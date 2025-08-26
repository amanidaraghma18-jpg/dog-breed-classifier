# Dog Breed Classifier - Udacity Project
# Author: Amani Daraghma

import torch
from torchvision import models, transforms
from PIL import Image
import os

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)

def predict(image_path, model, idx_to_class):
    image = process_image(image_path)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, pred_class = torch.max(output, 1)
    return idx_to_class[pred_class.item()]

if __name__ == "__main__":
    model = models.resnet18(pretrained=True)
    idx_to_class = {0: "Beagle", 1: "German Shepherd", 2: "Golden Retriever"}
    image_folder = "sample_images"

    for file in os.listdir(image_folder):
        if file.endswith((".jpg", ".png")):
            path = os.path.join(image_folder, file)
            prediction = predict(path, model, idx_to_class)
            print(f"{file} ---> {prediction}")
