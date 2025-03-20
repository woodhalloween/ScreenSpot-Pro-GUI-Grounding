import os
from PIL import Image
import base64
from io import BytesIO
import requests
import numpy as np

def process_vision_info(messages):
    """Process vision information from the messages.
    
    Args:
        messages: Messages with potential image data
        
    Returns:
        image_inputs: List of processed images
        video_inputs: List of processed videos (empty for now)
    """
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        if message["role"] == "user":
            for content in message["content"]:
                if content["type"] == "image":
                    image_data = content["image"]
                    if isinstance(image_data, str):
                        if image_data.startswith("http://") or image_data.startswith("https://"):
                            # Handle image URLs
                            response = requests.get(image_data)
                            image = Image.open(BytesIO(response.content))
                        elif os.path.exists(image_data) and os.path.isfile(image_data):
                            # Handle local file paths
                            image = Image.open(image_data)
                        elif image_data.startswith("data:image/"):
                            # Handle base64 encoded images
                            image_data = image_data.split(",")[1]
                            image = Image.open(BytesIO(base64.b64decode(image_data)))
                        else:
                            # Skip invalid image data
                            continue
                    elif isinstance(image_data, Image.Image):
                        # Already a PIL Image
                        image = image_data
                    else:
                        # Skip invalid image data
                        continue
                    
                    # Convert to RGB if needed
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    
                    image_inputs.append(image)
    
    return image_inputs, video_inputs 