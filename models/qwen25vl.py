import torch
import json
import re
import os
import tempfile
from PIL import Image
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import NousFnCallPrompt, Message, ContentItem
from utils.device_interaction_tools import ComputerUse

def bbox_from_point(point, size=0.05):
    """ポイント座標からBBoxを作成"""
    x, y = point
    half_size = size / 2
    return [x - half_size, y - half_size, x + half_size, y + half_size]

def image_to_temp_filename(image):
    """画像を一時ファイルとして保存"""
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name

class Qwen25VLModel:
    def load_model(self, model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct"):
        """モデルのロード"""
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_name_or_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.model.eval()
        print("Model loaded successfully.")

    def set_generation_config(self, temperature=0, max_new_tokens=256, **kwargs):
        """生成設定を更新する
        
        Args:
            temperature: 生成時の温度パラメータ
            max_new_tokens: 生成する最大トークン数
            **kwargs: その他の生成設定パラメータ
        """
        # デフォルト値を設定
        generation_config = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        
        # 追加のパラメータで更新
        generation_config.update(kwargs)
        
        # クラス変数として保存
        self.generation_config = generation_config
        print(f"Generation config set: {generation_config}")

    def ground_only_positive(self, instruction, image):
        """正の結果のみを処理（ターゲットが存在すると仮定）"""
        # 画像のパスを取得
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
        
        # 画像サイズを取得
        input_image = Image.open(image_path)
        resized_height, resized_width = smart_resize(
            input_image.height,
            input_image.width,
            factor=self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
            min_pixels=self.processor.image_processor.min_pixels,
            max_pixels=self.processor.image_processor.max_pixels,
        )
        
        # コンピュータ使用関数の初期化
        computer_use = ComputerUse(
            cfg={"display_width_px": resized_width, "display_height_px": resized_height}
        )
        
        # プロンプトの作成
        prompt = f"Output the bounding box in the image corresponding to the instruction \"{instruction}\" with grounding."
        
        # メッセージ準備
        nous_prompt = NousFnCallPrompt()
        message = nous_prompt.preprocess_fncall_messages(
            messages=[
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
                Message(role="user", content=[
                    ContentItem(text=prompt),
                    ContentItem(image=f"file://{image_path}")
                ]),
            ],
            functions=[computer_use.parameters],
            lang=None,
        )
        message = [msg.model_dump() for msg in message]
        
        # 入力処理
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[input_image], padding=True, return_tensors="pt").to(self.model.device)
        
        # 生成
        generation_params = {"max_new_tokens": 2048}
        if hasattr(self, "generation_config"):
            generation_params.update(self.generation_config)
        output_ids = self.model.generate(**inputs, **generation_params)
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        
        # 結果の解析
        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": output_text,
            "bbox": None,
            "point": None
        }
        
        # ツール呼び出しの解析
        if '<tool_call>' in output_text and '</tool_call>' in output_text:
            try:
                tool_call_text = output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0]
                action = json.loads(tool_call_text)
                
                if 'arguments' in action and 'coordinate' in action['arguments']:
                    coordinate = action['arguments']['coordinate']
                    # 座標を正規化（0-1範囲）
                    normalized_coordinate = [
                        coordinate[0] / resized_width,
                        coordinate[1] / resized_height
                    ]
                    
                    # BBoxの作成（簡易的な実装）
                    bbox_size = 0.05  # サイズは調整可能
                    bbox = [
                        max(0, normalized_coordinate[0] - bbox_size), 
                        max(0, normalized_coordinate[1] - bbox_size),
                        min(1, normalized_coordinate[0] + bbox_size), 
                        min(1, normalized_coordinate[1] + bbox_size)
                    ]
                    
                    result_dict["bbox"] = bbox
                    result_dict["point"] = normalized_coordinate
            except Exception as e:
                print(f"Error parsing tool call: {e}")
                print(output_text)
        
        return result_dict

    def ground_allow_negative(self, instruction, image):
        """ネガティブな結果も処理（ターゲットが存在しない可能性）"""
        # 画像のパスを取得
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
        
        # 画像サイズを取得
        input_image = Image.open(image_path)
        resized_height, resized_width = smart_resize(
            input_image.height,
            input_image.width,
            factor=self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
            min_pixels=self.processor.image_processor.min_pixels,
            max_pixels=self.processor.image_processor.max_pixels,
        )
        
        # コンピュータ使用関数の初期化
        computer_use = ComputerUse(
            cfg={"display_width_px": resized_width, "display_height_px": resized_height}
        )
        
        # プロンプトの作成
        prompt = f'Output the bounding box in the image corresponding to the instruction "{instruction}". If the target does not exist, respond with "Target does not exist".'
        
        # メッセージ準備
        nous_prompt = NousFnCallPrompt()
        message = nous_prompt.preprocess_fncall_messages(
            messages=[
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
                Message(role="user", content=[
                    ContentItem(text=prompt),
                    ContentItem(image=f"file://{image_path}")
                ]),
            ],
            functions=[computer_use.parameters],
            lang=None,
        )
        message = [msg.model_dump() for msg in message]
        
        # 入力処理
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[input_image], padding=True, return_tensors="pt").to(self.model.device)
        
        # 生成
        generation_params = {"max_new_tokens": 2048}
        if hasattr(self, "generation_config"):
            generation_params.update(self.generation_config)
        output_ids = self.model.generate(**inputs, **generation_params)
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        
        # 結果の解析
        result_dict = {
            "result": None,
            "format": "x1y1x2y2",
            "raw_response": output_text,
            "bbox": None,
            "point": None
        }
        
        # ツール呼び出しの解析
        if '<tool_call>' in output_text and '</tool_call>' in output_text:
            try:
                tool_call_text = output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0]
                action = json.loads(tool_call_text)
                
                if 'arguments' in action and 'coordinate' in action['arguments']:
                    coordinate = action['arguments']['coordinate']
                    # 座標を正規化（0-1範囲）
                    normalized_coordinate = [
                        coordinate[0] / resized_width,
                        coordinate[1] / resized_height
                    ]
                    
                    # BBoxの作成（簡易的な実装）
                    bbox_size = 0.05  # サイズは調整可能
                    bbox = [
                        max(0, normalized_coordinate[0] - bbox_size), 
                        max(0, normalized_coordinate[1] - bbox_size),
                        min(1, normalized_coordinate[0] + bbox_size), 
                        min(1, normalized_coordinate[1] + bbox_size)
                    ]
                    
                    result_dict["bbox"] = bbox
                    result_dict["point"] = normalized_coordinate
                    result_dict["result"] = "positive"
            except Exception as e:
                print(f"Error parsing tool call: {e}")
                print(output_text)
        elif "Target does not exist".lower() in output_text.lower():
            result_dict["result"] = "negative"
        else:
            result_dict["result"] = "wrong_format"
        
        return result_dict