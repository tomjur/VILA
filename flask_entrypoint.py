import datetime
import os
from dataclasses import dataclass

import torch
from flask import Flask, request, jsonify, send_file
import numpy as np
import base64
import io
from typing import List, Tuple, Optional

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.conversation import conv_templates, SeparatorStyle

app = Flask(__name__)


@dataclass
class VilaInputs:
    input_ids: torch.Tensor
    images_tensor: Optional[torch.Tensor]
    stop_str: str
    stopping_criteria: object


class ModelWrapper:
    def __init__(self, model_path="Efficient-Large-Model/VILA1.5-3b", model_base=None, conv_mode='vicuna_v1'):
        self.model_path = model_path
        self.model_base = model_base
        self.conv_mode = conv_mode

        self.hidden_layer_to_use_as_embedding = -1

        disable_torch_init()
        self.model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path, self.model_name, self.model_base)
        self.device = self.model.device

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        self.conv_mode = conv_mode

        # warm up the model by doing a single query
        self.predict_completion([[('text', 'Hello')]])
        print("done warming up model")

    def predict_completion(self, query_parts: List[Tuple[str, str]], temperature=0.2, top_p=None, num_beams=1, max_new_tokens=512) -> str:
        print('predict_completion called')
        result = []
        for vila_inputs in self._compose_model_inputs(query_parts):
            completion, _, _ = self._run_vila(vila_inputs, temperature, top_p, num_beams, max_new_tokens)
            result.append(completion)
        return result

    def predict_representation_base(self, query_parts: List[Tuple[str, str]]) -> np.array:
        print('predict_representation_base called')
        result = []
        for vila_inputs in self._compose_model_inputs(query_parts):
            _, base_hidden_states, _ = self._run_vila(vila_inputs, 0, None, 1, 1)
            result.append(base_hidden_states)
        result = torch.stack(result)
        return result

    def predict_representation_after_completion(self, query_parts: List[Tuple[str, str]], temperature=0.2, top_p=None, num_beams=1, max_new_tokens=512) -> np.array:
        print('predict_representation_after_completion called')
        result = []
        for vila_inputs in self._compose_model_inputs(query_parts):
            _, _, post_generation_hidden_states = self._run_vila(vila_inputs, temperature, top_p, num_beams, max_new_tokens)
            result.append(post_generation_hidden_states.cpu().numpy())
        result = torch.stack(result)
        return result

    def _compose_model_inputs(self, query_parts: List[Tuple[str, str]]) -> list[VilaInputs]:
        conv_template = conv_templates[self.conv_mode]
        stop_str = conv_template.sep if conv_template.sep_style != SeparatorStyle.TWO else conv_template.sep2
        keywords = [stop_str]

        query = ["" for _ in range(len(query_parts))]
        images_tensor = []
        input_ids = []
        stopping_criteria = []
        for env_id in range(len(query_parts)):
            images = []
            for part_type, part in query_parts[env_id]:
                if part_type == 'text':
                    query[env_id] += f" {part}"
                elif part_type == 'image':
                    query[env_id] += f" {DEFAULT_IMAGE_TOKEN}"
                    images.append(part)

            images_tensor.append(self._process_images(images))
            conv = conv_template.copy()
            conv.append_message(conv.roles[0], query[env_id])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids.append(tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                                   return_tensors="pt").unsqueeze(0).cuda())

            stopping_criteria.append(KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids[env_id]))

        return [VilaInputs(_input_ids, _images_tensor, stop_str, _stopping_criteria)
                for _input_ids, _images_tensor, _stopping_criteria, _query in zip(input_ids, images_tensor, stopping_criteria, query)]

    @staticmethod
    def _process_images(images: List[torch.Tensor]) -> Optional[torch.Tensor]:
        if len(images) == 0:
            return None
        images = torch.stack(images)
        assert len(images.shape) == 4, 'images should be of shape (num_images, height, width, channels)'
        assert images.shape[1] == images.shape[2] == 384, 'images should be square, of size 384x384'
        assert images.shape[3] == 3, 'images should have 3 channels (RGB)'
        # rescale: 0-255 -> 0-1
        images = images / 255.0
        # normalize around 0
        image_mean = torch.tensor([0.5] * 3, device=images.device)
        image_std = torch.tensor([0.5] * 3, device=images.device)
        images = (images - image_mean) / image_std
        # set the first dimension to be channels
        images = images.permute(0, 3, 1, 2)
        # convert to float16
        images = images.type(torch.float16)
        return images

    def _run_vila(self, inputs: VilaInputs, temperature, top_p, num_beams, max_new_tokens):
        with torch.inference_mode():
            generation_output = self.model.generate(
                inputs.input_ids,
                images=[inputs.images_tensor,] if inputs.images_tensor is not None else None,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[inputs.stopping_criteria],
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_ids = generation_output.sequences
            base_hidden_states = generation_output.hidden_states[0][self.hidden_layer_to_use_as_embedding][:, -1, :]
            post_generation_hidden_states = generation_output.hidden_states[-1][self.hidden_layer_to_use_as_embedding][:, -1, :]

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(inputs.stop_str):
            outputs = outputs[: -len(inputs.stop_str)]
        outputs = outputs.strip()

        return outputs, base_hidden_states, post_generation_hidden_states


model_wrapper = ModelWrapper()


def _image_tensor_from_file(file) -> torch.Tensor:
    buffer = io.BytesIO(file.read())
    tensor = torch.load(buffer)
    return tensor.to('cuda' if torch.cuda.is_available() else 'cpu')


@app.route('/predict_completion', methods=['POST'])
def call_predict_completion():
    query_parts = []
    query_text = request.form['text'] if 'text' in request.form else None
    query_parts.append(('text', query_text))
    temperature = float(request.form['temperature']) if 'temperature' in request.form else None
    top_p = float(request.form['top_p']) if 'top_p' in request.form else None
    num_beams = int(request.form['num_beams']) if 'num_beams' in request.form else None
    max_new_tokens = int(request.form['max_new_tokens']) if 'max_new_tokens' in request.form else None
    result = model_wrapper.predict_completion(query_parts, temperature, top_p, num_beams, max_new_tokens)
    return jsonify(result)


@app.route('/predict_representation', methods=['POST'])
def call_predict_representation_base():
    # Initialize a list to store query parts
    query_parts = [[] for _ in range(len(request.files))]
    query_text = request.form.getlist('text')

    # Iterate over the files in the request
    for env_id, (key, file) in enumerate(request.files.items()):
        # Deserialize the file into a tensor
        tensor = _image_tensor_from_file(file)

        # Unbind the tensor into individual images along the first dimension
        unbound_images = tensor.unbind(dim=0)

        # Add each unbound image to the query parts
        for image in unbound_images:
            query_parts[env_id].append(('image', image))
        query_parts[env_id].append(('text', query_text[env_id]))

    result = model_wrapper.predict_representation_base(query_parts)

    # Assuming model_wrapper is already defined and loaded
    if torch.any(torch.isnan(result)):
        # print error:
        os.makedirs("errors", exist_ok=True)
        print("Error: NaNs in the result tensor, query parts:", query_parts)
        # save query parts to a local file, the name of the file is a formatted time string:
        file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pt"
        torch.save(query_parts, os.path.join('errors', file_name))


    # Convert the result tensor to bytes
    buffer = io.BytesIO()
    torch.save(result, buffer)
    buffer.seek(0)

    # Send the serialized tensor as a file to the client
    return send_file(buffer, as_attachment=True, download_name='result.pt', mimetype='application/octet-stream')


@app.route('/predict_representation_after_completion', methods=['POST'])
def call_predict_representation_after_completion():
    query_parts = []
    query_text = request.form['text'] if 'text' in request.form else None
    query_parts.append(('text', query_text))
    temperature = float(request.form['temperature']) if 'temperature' in request.form else None
    top_p = float(request.form['top_p']) if 'top_p' in request.form else None
    num_beams = int(request.form['num_beams']) if 'num_beams' in request.form else None
    max_new_tokens = int(request.form['max_new_tokens']) if 'max_new_tokens' in request.form else None
    result = model_wrapper.predict_representation_after_completion(query_parts, temperature, top_p, num_beams, max_new_tokens)
    return jsonify(result.tolist())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    print("app closed.")
