import json
import os
import re
import tempfile
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import torch

import openai

from PIL import Image, ImageDraw
import ast


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# bbox -> point (str)
def bbox_2_point(bbox, dig=2):
    # bbox [left, top, right, bottom]
    point = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    point = [f"{item:.2f}" for item in point]
    point_str = "({},{})".format(point[0], point[1])
    return point_str

# bbox -> bbox (str)
def bbox_2_bbox(bbox, dig=2):
    bbox = [f"{item:.2f}" for item in bbox]
    bbox_str = "({},{},{},{})".format(bbox[0], bbox[1], bbox[2], bbox[3])
    return bbox_str

# point (str) -> point
def pred_2_point(s):
    floats = re.findall(r'-?\d+\.?\d*', s)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        return floats
    elif len(floats) == 4:
        return [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
    else:
        return None

# bbox (qwen str) -> bbox
import re

def extract_bbox(s):
    # Extract text between <|box_start|> and <|box_end|> tags
    match = re.search(r'<\|box_start\|\>(.*?)<\|box_end\|\>', s)
    
    if match:
        # Get the text between the tags
        extracted_text = match.group(1)
        
        # Remove parentheses and brackets
        cleaned_text = re.sub(r'[()\[\]]', '', extracted_text)
        
        # Extract four numbers from the cleaned text
        pattern = r"(\d+),\s*(\d+),\s*(\d+),\s*(\d+)"
        numbers = re.findall(pattern, cleaned_text)
        
        if numbers:
            # Return the first match as tuples of integers
            x1, y1, x2, y2 = numbers[0]
            return (int(x1), int(y1)), (int(x2), int(y2))
    
    return None



def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name




class AriaUIModel():
    def load_model(self, model_name_or_path="Aria-UI/Aria-UI-base", device="cuda"):
        self.device = device
        self.model = model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        # Setting default generation config
        self.generation_config = GenerationConfig.from_pretrained("Aria-UI/Aria-UI-base", trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_length=4096,
            do_sample=False,
            temperature=0.0
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        # self.model.generation_config = GenerationConfig(**self.generation_config)


    def ground_only_positive(self, instruction, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
        
        prompt_origin = 'Given a GUI image, what are the relative (0-1000) pixel point coordinates for the element corresponding to the following instruction or description: '
        full_prompt = prompt_origin + instruction
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": None, "type": "image"},
                    {"text": full_prompt, "type": "text"},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = self.model.generate(
                **inputs,
                max_new_tokens=50,
                stop_strings=["<|im_end|>"],
                tokenizer=self.processor.tokenizer,
                do_sample=False,
                # temperature=0.9,
            )

        output_ids = output[0][inputs["input_ids"].shape[1] :]
        response = self.processor.decode(output_ids, skip_special_tokens=True)
        print(response)

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        try:
            point = ast.literal_eval(response.replace("<|im_end|>", "").replace("```", "").replace(" ", "").strip())
            x, y = point
            click_point = [x / 1000, y / 1000]
            if 0 <= x <= 1 and 0 <= y <= 1:
                result_dict["point"] = click_point
                print(click_point)
        except Exception as e:
            point = None

        return result_dict

class AriaUIVLLMModel():
    def load_model(self, model_name_or_path="Aria-UI/Aria-UI-base"):
        from vllm import LLM, SamplingParams
        self.sampling_params = SamplingParams(
            max_tokens=50,
            top_k=1,
            stop=["<|im_end|>"],
            temperature=0
        )

        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_name_or_path, 
                            trust_remote_code=True, 
                            use_fast=False
                        )
        self.model = LLM(
            model=model_name_or_path,
            tokenizer_mode="slow",
            dtype="bfloat16",
            trust_remote_code=True,
        )

    def set_generation_config(self, **kwargs):
        # self.sampling_params = SamplingParams(**kwargs)
        # TODO: fix
        pass


    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."

        prompt_origin = 'Given a GUI image, what are the relative (0-1000) pixel point coordinates for the element corresponding to the following instruction or description: '
        full_prompt = prompt_origin + instruction

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": full_prompt,
                    }
                ],
            }
        ]

        message = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        outputs = self.model.generate(
            {
                "prompt_token_ids": message,
                "multi_modal_data": {
                    "image": [
                        image,
                    ],
                    # "max_image_size": 980,  # [Optional] The max image patch size, default `980`
                    "split_image": True,  # [Optional] whether to split the images, default `True`
                },
            },
            sampling_params=self.sampling_params,
        )
        for o in outputs:
            generated_tokens = o.outputs[0].token_ids
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        try:
            point = ast.literal_eval(response.replace("<|im_end|>", "").replace("```", "").replace(" ", "").strip())
            x, y = point
            x = x / 1000
            y = y / 1000
            if 0 <= x <= 1 and 0 <= y <= 1:
                result_dict["point"] = [x, y]
        except Exception as e:
            pass

        return result_dict
