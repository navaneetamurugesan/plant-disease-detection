import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import CNN  # Ensure CNN is defined correctly

# Define transformations for dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load dataset (Ensure correct dataset path)
data_dir = 'dataset_path'  # Change this to actual dataset path
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Get number of classes
disease_classes = train_dataset.classes  # Get class names
num_classes = len(disease_classes)
print(f"Number of disease classes: {num_classes}")

# Initialize model
model = CNN.CNN(3)
model.dense_layers[4] = nn.Linear(1024, num_classes)  # Modify output layer to match number of classes

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# Save trained model
torch.save(model.state_dict(), 'Plant_Disease_Detection_Model.pth')
print("✅ Model training complete and saved successfully.")
