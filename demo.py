import gradio as gr
from gradio_imageslider import ImageSlider
import tempfile
import numpy as np
from PIL import Image
import json
import random
import io
import uuid
import os
import cv2

from services.clarity_img_enhancer import clarity_img_enhancer
from services.generate_img_with_flux import generate_image_with_flux_lora
import httpx
import time

from inferences import Predict
from data.data_process import split_digit_from_img, makeContours
from dotenv import load_dotenv
load_dotenv()
predict = Predict()
def image_enhance_clarity():
    gr.Markdown("## Image Enhancement with Clarity")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            run_button = gr.Button(value="Enhance Image")
        with gr.Column():
            output_slider = ImageSlider(label="Before / After")

    with gr.Accordion("Advanced Options", open=False):
        prompt = gr.Textbox(
            label="Prompt",
            placeholder="masterpiece, best quality, highres, <lora:more_details:0.5> <lora:SDXLrender_v2.0:1>",
        )
        negative_prompt = gr.Textbox(
            label="Negative Prompt",
            placeholder="(worst quality, low quality, normal quality:2) JuggernautNegative-neg",
        )
        seed = gr.Slider(
            minimum=0,
            maximum=10_000,
            value=1337,
            step=1,
            label="Seed",
        )
        scale_factor = gr.Slider(
            minimum=1,
            maximum=4,
            value=2,
            step=0.2,
            label="Scale Factor",
        )
        num_inference_steps = gr.Slider(
            minimum=1,
            maximum=30,
            value=18,
            step=1,
            label="Number of Inference Steps",
        )

    def enhance_image(image, steps, prompt, scale_factor, negative_prompt):
        assetID = str(uuid.uuid4())
        return clarity_img_enhancer(image, steps, prompt, scale_factor, negative_prompt, assetID)

    run_button.click(
        fn=enhance_image,
        inputs=[
            input_image,
            num_inference_steps,
            prompt,
            scale_factor,
            negative_prompt,
        ],
        outputs=output_slider,
    )

def predict_uploaded_image(image):
    img = Image.fromarray(image.astype('uint8'), 'RGB')
    
    # Save image to path
    path = {
        "images": "data/test/digit_test.png",
        "lines": "data/lines/",
        "words": "data/words/"
    }
    img.save(path["images"])
    
    results = split_digit_from_img(path)
    
    numbers = ""
    for img in results:
        _, binary = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
        binary = np.pad(binary, (25, 25), "constant", constant_values=(0, 0))
        img = cv2.resize(binary, (28, 28))
        cv2.imwrite("data/test/digittest.png", img)
        res = predict(img)
        digit = int(res.argmax())
        numbers += str(digit)
    
    return numbers

def predict_drawn_digit(image):
    drawing_layer = image['layers'][0]
    print(drawing_layer)
    print(image )
    # Save image to path
    path = "Test.png"
    cv2.imwrite(path, drawing_layer)
    img = cv2.cvtColor(drawing_layer.astype('uint8'), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (28, 28))
    res = predict(img)
    digit = int(res.argmax())
    return f"Predicted Digit: {digit}"

def digit_recognizer():
    gr.Markdown("## Digit Recognizer")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            recognize_button = gr.Button(value="Recognize Digit")
        with gr.Column():
            result_text = gr.Textbox(label="Recognized Digits", interactive=False)

    recognize_button.click(
        fn=predict_uploaded_image,
        inputs=[input_image],
        outputs=[result_text],
    )

    gr.Markdown("## Draw a Digit")
    with gr.Row():
        with gr.Column():
            canvas = gr.Sketchpad(label="Draw here")
            draw_button = gr.Button(value="Recognize Drawn Digit")
        with gr.Column():
            draw_result_text = gr.Textbox(label="Recognized Digit", interactive=False)

    draw_button.click(
        fn=predict_drawn_digit,
        inputs=[canvas],
        outputs=[draw_result_text],
    )

def image_generator_flux():
    with open('services/loras.json') as f:
        loras = json.load(f)

    MAX_SEED = 2**32 - 1

    def update_selection(evt: gr.SelectData, width, height):
        selected_lora = loras[evt.index]
        new_placeholder = f"Type a prompt for {selected_lora['title']}"
        lora_repo = selected_lora["repo"]
        updated_text = f"### Selected: [{lora_repo}](https://huggingface.co/{lora_repo}) ✨"
        if "aspect" in selected_lora:
            if selected_lora["aspect"] == "portrait":
                width = 768
                height = 1024
            elif selected_lora["aspect"] == "landscape":
                width = 1024
                height = 768
        return (
            gr.update(placeholder=new_placeholder),
            updated_text,
            evt.index,
            width,
            height,
        )

    def run_lora(prompt, steps, selected_index, randomize_seed, seed, width, height, output_format, output_quality, noI):
        if selected_index is None:
            raise gr.Error("You must select a LoRA before proceeding.")

        selected_lora = loras[selected_index]
        hf_lora = selected_lora["repo"]
        trigger_word = selected_lora["trigger_word"]

        if randomize_seed:
            seed = random.randint(0, MAX_SEED)

        assetID = f"demo_{seed}"  # You might want to generate a unique ID here

        local_paths = []
        remote_urls = []
        for _ in range(noI):
            local_path, remote_url = generate_image_with_flux_lora(
                prompt=prompt,
                trigger_word=trigger_word,
                steps=steps,
                width=width,
                height=height,
                hf_lora=hf_lora,
                output_format=output_format,
                num_inference_steps=steps,
                assetID=assetID,
                output_quality=output_quality
            )
            local_paths.append(local_path)
            remote_urls.append(remote_url)

        return local_paths, seed, remote_urls

    gr.Markdown("## Image Generation with Flux LoRA")
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="Prompt", lines=1, placeholder="Type a prompt after selecting a LoRA")
        with gr.Column(scale=1):
            generate_button = gr.Button("Generate", variant="primary")

    with gr.Row():
        with gr.Column(scale=3):
            selected_info = gr.Markdown("")
            gallery = gr.Gallery(
                [(item["image"], item["title"]) for item in loras],
                label="LoRA Gallery",
                allow_preview=False,
                columns=3
            )

        with gr.Column(scale=4):
            result = gr.Gallery(label="Generated Image")
            # Display the remote URLs of the generated images
            remote_urls = gr.Textbox(label="Remote URLs", interactive=False, lines=5)

    with gr.Row():
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Column():
                with gr.Row():
                    steps = gr.Slider(label="Steps", minimum=1, maximum=50, step=1, value=28)

                with gr.Row():
                    width = gr.Slider(label="Width", minimum=256, maximum=1536, step=64, value=1024)
                    height = gr.Slider(label="Height", minimum=256, maximum=1536, step=64, value=1024)

                with gr.Row():
                    randomize_seed = gr.Checkbox(True, label="Randomize seed")
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0, randomize=True)

                with gr.Row():
                    output_format = gr.Dropdown(["png", "jpg", "webp"], label="Output Format", value="png")
                    output_quality = gr.Slider(label="Output Quality", minimum=1, maximum=100, step=1, value=75)

                with gr.Row():
                    noI = gr.Slider(label="Number of Images", minimum=1, maximum=5, step=1, value=1)

    selected_index = gr.State(None)

    gallery.select(
        update_selection,
        inputs=[width, height],
        outputs=[prompt, selected_info, selected_index, width, height]
    )

    generate_button.click(
        fn=run_lora,
        inputs=[prompt, steps, selected_index, randomize_seed, seed, width, height, output_format, output_quality, noI],
        outputs=[result, seed, remote_urls]
    )

def degit_predict():
    digit_recognizer()

if __name__ == "__main__":
    title = r"""
            <div style="text-align: center;">
                <h1> AI Image Tools </h1>
                </br>
            </div>
        """
    custom_css = "footer {visibility: hidden}"
    block = gr.Blocks(title="Cloud-computing mid-term - Vuong Tuan Cuong", css=custom_css).queue()
    with block:
        gr.HTML(title)

        with gr.TabItem("Image Generation with Flux LoRA"):
            image_generator_flux()

        with gr.TabItem("Image Enhancement with Clarity"):
            image_enhance_clarity()

        with gr.TabItem("Digit Recognizer"):
            digit_recognizer()

    block.launch(share=True)  # Set timeout to 300 seconds

    # Load environment variables
