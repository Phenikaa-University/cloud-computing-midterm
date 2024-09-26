import torch
from model.model import CNN
from torchvision import transforms
import cv2
from data.data_process import makeContours

SAVE_MODEL_PATH = "checkpoints/best_accuracy.pth"

class Predict():
    def __init__(self):
        device = torch.device("cpu")
        self.model = CNN().to(device)
        self.model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=device))
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


    def __call__(self, img):
        img = cv2.resize(img, (28, 28))
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)  

        self.model.eval()
        with torch.no_grad():
            preds = self.model(tensor)
            preds = preds.detach().numpy()[0]

        return preds
    
