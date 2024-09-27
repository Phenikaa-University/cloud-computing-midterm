import requests
import replicate
import os
from dotenv import load_dotenv
load_dotenv()

from services.utils import calculateDuration

def generate_image_with_flux_lora(prompt: str, trigger_word: str, steps: int, width: int, height: int, hf_lora: str, output_format: str, num_inference_steps: int, assetID: str, output_quality: int):
    with calculateDuration("Generate image with Flux Lora"):
        input = {
            "prompt": f"{prompt} {trigger_word}",
            "hf_lora": hf_lora,
            "aspect_ratio": "1:1", # 21:9, 16:9, 4:3, 1:1, 3:4, 9:16, 9:21
            "num_inference_steps": steps,
            "width": width,
            "height": height,
            "output_format": output_format,
            "output_quality": output_quality,
            "num_inference_steps": num_inference_steps,
            "lora_scale": 0.8,
            "num_outputs": 1,
            "guidance_scale": 3.5,
        }
        output_path = replicate.run(
            "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
            input
        )
    
    path = f'db/image/generation/{assetID}'
    if not os.path.exists(path):
        os.makedirs(path)
    # Save the image
    with open(f"{path}/image.{output_format}", "wb") as f:
        f.write(requests.get(output_path[0]).content)
    return (f"{path}/image.{output_format}", output_path[0])