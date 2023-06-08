import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import alexnet

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the paths to your training and validation datasets
train_data_path = "/home/woody/iwso/iwso092h/leaf_clf/train/train_split"
valid_data_path = "/home/woody/iwso/iwso092h/leaf_clf/train/test_split"

# Define the transformation for your images (resize, normalize, etc.)
image_transforms = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.RandomHorizontalFlip(),  # Data augmentation: randomly flip images horizontally
    transforms.RandomVerticalFlip(),  # Data augmentation: randomly flip images vertically
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4495, 0.4654, 0.4004], std=[0.1785, 0.1566, 0.1945])
])

# Load the training and validation datasets
train_dataset = datasets.ImageFolder(root=train_data_path, transform=image_transforms)
valid_dataset = datasets.ImageFolder(root=valid_data_path, transform=image_transforms)

# Compute class weights
train_targets = torch.tensor(train_dataset.targets)

class_counts = torch.bincount(train_targets)
total_samples = len(train_dataset)
class_weights = total_samples / (len(train_dataset) * class_counts)

class_weights = class_weights.to(device)

# Create data loaders to load the data in batches
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)

# Load the pre-trained AlexNet model
alexnet_model = alexnet(pretrained=True)

# Freeze all the model parameters
for param in alexnet_model.parameters():
    param.requires_grad = False

# Modify the last fully connected layer to match the number of classes in your dataset
num_classes = len(train_dataset.classes)
alexnet_model.classifier[6] = nn.Linear(4096, num_classes)

# Move the model to the device
alexnet_model = alexnet_model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(alexnet_model.parameters())

# Training loop
num_epochs = 30
save_interval = 5


for epoch in range(num_epochs):
    train_loss = 0.0
    valid_loss = 0.0

    # Train the model
    alexnet_model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = alexnet_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    # Validate the model
    alexnet_model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = alexnet_model(images)
            loss = criterion(outputs, labels)

            valid_loss += loss.item() * images.size(0)

    # Calculate average losses
    train_loss = train_loss / len(train_dataset)
    valid_loss = valid_loss / len(valid_dataset)

    # Print the progress
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
    if (epoch + 1) % save_interval == 0:
        checkpoint_path = f"/home/woody/iwso/iwso092h/leaf_clf/model_alex/model_epoch_{epoch+1}.pth"
        torch.save(alexnet_model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint at epoch {epoch+1}")
        
# Save the fine-tuned model
# torch.save(alexnet_model.state_dict(),
#               '/home/woody/iwso/iwso092h/leaf_clf/finetuned_model_gpt.pth')
