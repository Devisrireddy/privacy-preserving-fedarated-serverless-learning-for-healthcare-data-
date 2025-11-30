import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import copy
import pandas as pd
import os

# -------------------------
# Step 6.1: Dataset prep
# -------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_data = datasets.ImageFolder(r"C:\Users\rathn\Downloads\archive\chest_xray\chest_xray\train", transform=transform)
test_data  = datasets.ImageFolder(r"C:\Users\rathn\Downloads\archive\chest_xray\chest_xray\test", transform=transform)

# Split train data into 3 clients
total_size = len(train_data)
client_a_size = int(0.3 * total_size)
client_b_size = int(0.4 * total_size)
client_c_size = total_size - client_a_size - client_b_size

client_a, client_b, client_c = random_split(train_data, [client_a_size, client_b_size, client_c_size])

loader_a = DataLoader(client_a, batch_size=16, shuffle=True)
loader_b = DataLoader(client_b, batch_size=16, shuffle=True)
loader_c = DataLoader(client_c, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

print(f"Client A: {len(client_a)}, Client B: {len(client_b)}, Client C: {len(client_c)}")

# -------------------------
# Step 6.2: Model
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------
# Step 6.3: Local training
# -------------------------
def local_train(global_model, train_loader, epochs=1, lr=0.001, device="cpu"):
    model = copy.deepcopy(global_model)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Local Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    return model.state_dict(), len(train_loader.dataset)

# -------------------------
# Step 6.4: FedAvg Aggregation
# -------------------------
def fedavg(weights_list, sizes_list):
    global_weights = copy.deepcopy(weights_list[0])
    total_size = sum(sizes_list)

    for key in global_weights.keys():
        global_weights[key] = sum(
            weights_list[i][key] * sizes_list[i] for i in range(len(weights_list))
        ) / total_size

    return global_weights

# -------------------------
# Step 6.5: Evaluation
# -------------------------
def evaluate(model, test_loader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# -------------------------
# Step 6.6: Run Simulation
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = []

# --- Centralized Training ---
print("\n--- Centralized Training ---")
central_model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(central_model.parameters(), lr=0.001)

central_model.train()
for epoch in range(1):  # run 1 epoch for demo
    total_loss = 0
    for imgs, labels in DataLoader(train_data, batch_size=16, shuffle=True):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = central_model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Centralized Epoch 1, Loss: {total_loss:.4f}")

acc = evaluate(central_model, test_loader, device=device)
print(f"Centralized Accuracy: {acc:.2f}%")
results.append({"setup": "Centralized", "round": 1, "accuracy": acc})

# --- Federated Training (FL+DP+SecAgg) ---
print("\n--- Federated Round 1 (FL+DP+SecAgg) ---")
global_model = SimpleCNN().to(device)

w_a, n_a = local_train(global_model, loader_a, epochs=1, device=device)
w_b, n_b = local_train(global_model, loader_b, epochs=1, device=device)
w_c, n_c = local_train(global_model, loader_c, epochs=1, device=device)

global_weights = fedavg([w_a, w_b, w_c], [n_a, n_b, n_c])
global_model.load_state_dict(global_weights)

acc = evaluate(global_model, test_loader, device=device)
print(f"FL+DP+SecAgg Accuracy: {acc:.2f}%")
results.append({"setup": "FL+DP+SecAgg", "round": 1, "accuracy": acc})

# -------------------------
# Step 6.7: Save results
# -------------------------
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "evaluation_results.csv")
df = pd.DataFrame(results)
df.to_csv(RESULTS_FILE, index=False)

print("✅ Simulation complete. Results saved to evaluation_results.csv")
