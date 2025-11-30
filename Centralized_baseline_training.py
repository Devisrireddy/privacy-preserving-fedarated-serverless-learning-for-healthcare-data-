import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 2.1: Transform (resize + convert to tensor + normalize)
transform = transforms.Compose([
    transforms.Resize((128,128)),   
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  
])

# Step 2.2: Load dataset

train_data = datasets.ImageFolder(r"E:\archive\chest_xray\chest_xray\train", transform=transform)
test_data  = datasets.ImageFolder(r"E:\archive\chest_xray\chest_xray\test", transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True,num_workers=0)
test_loader  = DataLoader(test_data, batch_size=16, shuffle=False,num_workers=0)

# Step 2.3: Define a very small CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # after pooling twice (128x128 → 32x32)
        self.fc2 = nn.Linear(128, 2)  # binary classification (Normal vs Pneumonia)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Step 2.4: Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 2.5: Training loop 
for epoch in range(2):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Step 2.6: Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
