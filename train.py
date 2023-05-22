import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms, datasets
import pandas as pd

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the images to a consistent size
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image tensors
])

train_data = datasets.ImageFolder('/home/woody/iwso/iwso092h/leaf_clf/train/train_split', transform=transform)
test_data = datasets.ImageFolder('/home/woody/iwso/iwso092h/leaf_clf/train/test_split', transform=transform)

batch_size = 32  

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = models.resnet50(pretrained=True)

num_classes = len(train_data.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda")
model.to(device)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_data)
    print(f"Epoch {epoch+1}/{num_epochs} - Training loss: {epoch_loss:.4f}")
    
    save_path = '/home/woody/iwso/iwso092h/leaf_clf/models/resnet_{}'.format(epoch)

    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, save_path)

    print("Model saved successfully.")









