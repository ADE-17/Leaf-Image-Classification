import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms, datasets
import pandas as pd
from PIL import Image
import os

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the images to a consistent size
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image tensors
])

device = torch.device("cuda")

model = models.resnet50(pretrained=True)

model.fc = nn.Linear(model.fc.in_features, 10)

test_data = datasets.ImageFolder('/home/woody/iwso/iwso092h/leaf_clf/train/test_split', transform=transform)

batch_size = 32  

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

saved_model_path = '/home/woody/iwso/iwso092h/leaf_clf/models/resnet_19'  
checkpoint = torch.load(saved_model_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

# correct = 0
# total = 0

# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.cpu(), labels.cpu()
        
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
        
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# accuracy = 100 * correct / total
# print(f"Test accuracy: {accuracy:.2f}%")

test_image_path = "/home/woody/iwso/iwso092h/leaf_clf/test"
test_image_list = os.listdir(test_image_path)
pred = []
for img in test_image_list:
    image = Image.open(os.path.join(test_image_path,img))
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    pred.append(predicted.item())
    
df = pd.DataFrame()
df['Label'] = pred
df.to_csv('pred.csv', index=False)