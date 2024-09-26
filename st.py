"""Create streamlit app for predict ditgit from image"""

import streamlit as st
from PIL import Image
import requests
import cv2
from streamlit_drawable_canvas import st_canvas
import numpy as np

from inferences import Predict
from data.data_process import split_digit_from_img, makeContours

predict = Predict()

# Set title
st.title("Digit Recognizer")

# Upload image
st.write("Upload image")
image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Display image
if image is not None:
    img = Image.open(image)
    st.image(img, caption="Uploaded Image.", use_column_width=True)
    # Save image to path
    path = {
        "images": "data/test/digit_test.png",
        "lines": "data/lines/",
        "words": "data/words/"
    }
    img.save(path["images"])
    
    results = split_digit_from_img(path)

    # Predict image
    if st.button("Predict"):
        # Print predict result
        st.write("Predict result:")
        numbers = ""
        for img in results:
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
            binary = np.pad(binary, (25, 25), "constant", constant_values=(0, 0))
            img = cv2.resize(binary, (28, 28))
            cv2.imwrite("data/test/digittest.png", img)
            res = predict(img)
            digit = int(res.argmax())
            numbers += str(digit)
        st.write(numbers)
        
st.header("Draw a digit below")
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",  # Màu nền sẽ được vẽ
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=150,width=150,
    drawing_mode="freedraw",
    key="canvas",
)

# Nếu có vẽ trên canvas
if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = predict(img)
    digit = int(res.argmax())
    st.write(f"Predicted Digit: {digit}")