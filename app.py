import shutil
from fastapi import FastAPI, Form, HTTPException, Request, File, UploadFile
from typing import AsyncGenerator, NoReturn, Annotated
from fastapi.middleware.cors import CORSMiddleware
from inferences import Predict
from data.data_process import split_digit_from_img

predict = Predict()

app = FastAPI()

# Configure CORS
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def hello_world():
    return "Hello, World!"

@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

@app.post("/update_image")
def update_image(image: UploadFile):
    path = {
        "images": image.file,
        "lines": "data/lines/",
        "words": "data/words/"
    }
    # Save uploaded image to path
    with open(path["images"], "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

@app.post("/predict")
def predict_image(image: UploadFile):
    # Get path from uploaded image
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)