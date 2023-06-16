import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import alexnet
from PIL import Image
import glob
import csv

test_image_path = r"C:\Users\ADE17\Desktop\Masters\Projects\Leaf-Image-Classification\Data\train\00awnh6on14cwjukhh96.jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path to your saved model
model_path = r'C:\Users\ADE17\Desktop\Masters\Projects\Leaf-Image-Classification\Data\model_epoch_20.pth'

alexnet_model = alexnet(pretrained=False)
num_classes = 10  # Update with the number of classes in your dataset
alexnet_model.classifier[6] = nn.Linear(4096, num_classes)
alexnet_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
alexnet_model = alexnet_model.to(device)
alexnet_model.eval()

image_transforms = transforms.Compose([
    transforms.Resize((227, 227)),  # Resize the image
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.4495, 0.4654, 0.4004], std=[0.1785, 0.1566, 0.1945])
])

image = Image.open(test_image_path).convert("RGB")
image = image_transforms(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = alexnet_model(image)
    _, predicted = torch.max(output, 1)

predicted_class = predicted.item()

print(predicted_class)