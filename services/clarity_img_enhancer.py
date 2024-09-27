import requests
import replicate
import os
from dotenv import load_dotenv
load_dotenv()
from PIL import Image
import io

from services.utils import calculateDuration

def clarity_img_enhancer(image: Image.Image, steps: int, prompt: str, scale_factor: float, negative_prompt: str, assetID: str):
    # Save the image to a temporary file
    temp_image_path = f"temp_{assetID}.png"
    image.save(temp_image_path)
    
    with calculateDuration("Enhance image with Clarity"):
        output = replicate.run(
            "philz1337x/clarity-upscaler:dfad41707589d68ecdccd1dfa600d55a208f9310748e44bfe35b4a6291453d5e",
            input={
                "seed": 1337,
                "image": open(temp_image_path, "rb"),
                "prompt": prompt,
                "dynamic": 6,
                "scheduler": "DPM++ 3M SDE Karras",
                "creativity": 0.35,
                "resemblance": 0.6,
                "scale_factor": scale_factor,
                "negative_prompt": negative_prompt,
                "num_inference_steps": steps
            }
        )
    
    # Remove the temporary file
    os.remove(temp_image_path)
    
    path = f'db/image/enhancement/{assetID}'
    if not os.path.exists(path):
        os.makedirs(path)
    # Save the enhanced image
    output_format = "png"  # Assuming the output is always PNG
    enhanced_image_path = f"{path}/enhanced_image.{output_format}"
    with open(enhanced_image_path, "wb") as f:
        f.write(requests.get(output[0]).content)
    
    # Open the enhanced image
    enhanced_image = Image.open(enhanced_image_path)
    
    return (image, enhanced_image)