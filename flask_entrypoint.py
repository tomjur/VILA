from dataclasses import dataclass

import torch
from flask import Flask, request, jsonify
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
        self.predict_completion([('text', 'Hello')])
        print("done warming up model")

    def predict_completion(self, query_parts: List[Tuple[str, str]], temperature=0.2, top_p=None, num_beams=1, max_new_tokens=512) -> str:
        print('predict_completion called')
        vila_inputs, query = self._compose_model_inputs(query_parts)
        print('query:', query)
        completion, _, _ = self._run_vila(vila_inputs, temperature, top_p, num_beams, max_new_tokens)
        print('completion:', completion)
        return completion

    def predict_representation_base(self, query_parts: List[Tuple[str, str]]) -> np.array:
        print('predict_representation_base called')
        vila_inputs, query = self._compose_model_inputs(query_parts)
        print('query:', query)
        _, base_hidden_states, _ = self._run_vila(vila_inputs, 0, None, 1, 1)
        result = base_hidden_states.cpu().numpy()
        print('result:', result.tolist())
        return result

    def predict_representation_after_completion(self, query_parts: List[Tuple[str, str]], temperature=0.2, top_p=None, num_beams=1, max_new_tokens=512) -> np.array:
        print('predict_representation_after_completion called')
        vila_inputs, query = self._compose_model_inputs(query_parts)
        print('query:', query)
        _, _, post_generation_hidden_states = self._run_vila(vila_inputs, temperature, top_p, num_beams, max_new_tokens)
        result = post_generation_hidden_states.cpu().numpy()
        print('result:', result.tolist())
        return result

    def _compose_model_inputs(self, query_parts: List[Tuple[str, str]]) -> Tuple[VilaInputs, str]:
        query = ""
        images = []
        for part_type, part in query_parts:
            if part_type == 'text':
                query += f" {part}"
            elif part_type == 'image':
                images.append(self._image_tensor_from_part(part))
                query += f" {DEFAULT_IMAGE_TOKEN}"

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = self._process_images(images)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        return VilaInputs(input_ids, images_tensor, stop_str, stopping_criteria), query

    def _image_tensor_from_part(self, base64_string: str) -> torch.Tensor:
        base64_bytes = base64_string.encode('utf-8')
        bytes_data = base64.b64decode(base64_bytes)

        buffer = io.BytesIO(bytes_data)
        numpy_array = np.load(buffer)

        tensor = torch.from_numpy(numpy_array)
        tensor = tensor.to(self.device)
        tensor = tensor.type(torch.uint8)
        return tensor

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


@app.route('/predict_completion', methods=['POST'])
def call_predict_completion():
    data = request.get_json()
    query_parts = data.get('query_parts')
    temperature = data.get('temperature')
    top_p = data.get('top_p')
    num_beams = data.get('num_beams')
    max_new_tokens = data.get('max_new_tokens')
    result = model_wrapper.predict_completion(query_parts, temperature, top_p, num_beams, max_new_tokens)
    return jsonify(result)


@app.route('/predict_representation', methods=['POST'])
def call_predict_representation_base():
    data = request.get_json()
    query_parts = data.get('query_parts')
    result = model_wrapper.predict_representation_base(query_parts)
    return jsonify(result.tolist())


@app.route('/predict_representation_after_completion', methods=['POST'])
def call_predict_representation_after_completion():
    data = request.get_json()
    query_parts = data.get('query_parts')
    temperature = data.get('temperature')
    top_p = data.get('top_p')
    num_beams = data.get('num_beams')
    max_new_tokens = data.get('max_new_tokens')
    result = model_wrapper.predict_representation_after_completion(query_parts, temperature, top_p, num_beams, max_new_tokens)
    return jsonify(result.tolist())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    print("app closed.")
