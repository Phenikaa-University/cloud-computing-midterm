"""Evaluate the model on the test set"""
import torch
import torch.nn as nn
import torchvision

from model.model import CNN
from torchvision import transforms
import matplotlib.pyplot as plt

SAVE_MODEL_PATH = "checkpoints/best_accuracy.pth"

def evaluate(opt):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    testset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size)
    
    device = torch.device("cuda:0" if True and torch.cuda.is_available() else "cpu")
    print("device:", device)
    
    model = CNN().to(device)
    model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=device))
    
    # Using and plot confusion matrix to evaluate the model
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import seaborn as sns
    
    model.eval()
    all_preds = []
    all_labels = []
    for i, (imgs, labels) in enumerate(testloader):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        preds = torch.argmax(preds, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("./data/results/confusion_matrix.png")
    
    # Calculate classification report
    from sklearn.metrics import classification_report
    print(classification_report(all_labels, all_preds))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_seed", type=int, default=1111, help="random seed setting")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=30, help="number of epochs to train")
    parser.add_argument("--valid_interval", type=int, default=1, help="validation interval")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--use_gpu", action="store_true", help="use gpu if available")
    opt = parser.parse_args()
    print("args", opt)
    
    evaluate(opt)