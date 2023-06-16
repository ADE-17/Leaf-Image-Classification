import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import alexnet
from PIL import Image
import glob
import csv

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path to your saved model
model_path = '/home/woody/iwso/iwso092h/leaf_clf/model_alex/model_epoch_20.pth'

# Load the saved model
alexnet_model = alexnet(pretrained=False)
num_classes = 10  # Update with the number of classes in your dataset
alexnet_model.classifier[6] = nn.Linear(4096, num_classes)
alexnet_model.load_state_dict(torch.load(model_path))
alexnet_model = alexnet_model.to(device)
alexnet_model.eval()

# Define the transformation for your images (resize, normalize, etc.)
image_transforms = transforms.Compose([
    transforms.Resize((227, 227)),  # Resize the image
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.4495, 0.4654, 0.4004], std=[0.1785, 0.1566, 0.1945])
])

# Define the path to your new images
new_image_paths = glob.glob("/home/woody/iwso/iwso092h/leaf_clf/test/*")

# Make predictions for each new image
results = []
count = 1

for image_path in new_image_paths:
    image = Image.open(image_path).convert("RGB")
    image = image_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = alexnet_model(image)
        _, predicted = torch.max(output, 1)
    
    predicted_class = predicted.item()
    results.append((image_path, predicted_class))
    # print(f"Image: {image_path}, Predicted Class: {predicted_class}")
    print(count, '/', len(new_image_paths)+1)
    count = count + 1
csv_file = "/home/hpc/iwso/iwso092h/mycodedir/leaf_clf_codes/results.csv"

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image", "Predicted Class"])
    writer.writerows(results)

print("Results saved successfully in CSV file.")

