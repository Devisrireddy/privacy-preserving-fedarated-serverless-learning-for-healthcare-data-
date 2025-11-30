import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import copy

# -------------------------
# Step 5.1: Dataset prep
# -------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_data = datasets.ImageFolder(r"E:\archive\chest_xray\chest_xray\train", transform=transform)
test_data  = datasets.ImageFolder(r"E:\archive\chest_xray\chest_xray\test", transform=transform)

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
# Step 5.2: Model
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
# Step 5.3: Local training with DP + Secure Masking
# -------------------------
def local_train(global_model, train_loader, epochs=1, lr=0.001, device="cpu", noise_scale=0.01):
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

    # DP Noise
    weights = model.state_dict()
    noisy_weights = {}
    for key, value in weights.items():
        noise = torch.normal(0, noise_scale, size=value.shape).to(value.device)
        noisy_weights[key] = value + noise

    # Secure Aggregation Mask
    mask = {}
    for key, value in noisy_weights.items():
        mask[key] = torch.randn_like(value)  # random mask
        noisy_weights[key] = noisy_weights[key] + mask[key]  # add mask

    return noisy_weights, mask, len(train_loader.dataset)

# -------------------------
# Step 5.4: FedAvg with mask cancellation
# -------------------------
def fedavg(masked_weights, masks, local_sizes):
    """Aggregate masked updates and cancel masks"""
    # Start with deep copy of first client's weights
    global_weights = copy.deepcopy(masked_weights[0])
    total_size = sum(local_sizes)

    for key in global_weights.keys():
        # Sum masked weights
        global_weights[key] = sum(
            (masked_weights[i][key] * local_sizes[i] for i in range(len(masked_weights)))
        ) / total_size

        # Cancel masks (subtract average of masks)
        global_weights[key] -= sum(
            (masks[i][key] * local_sizes[i] for i in range(len(masks)))
        ) / total_size

    return global_weights

# -------------------------
# Step 5.5: Evaluation
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
# Step 5.6: Federated rounds
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_model = SimpleCNN().to(device)

num_rounds = 3
for rnd in range(1, num_rounds+1):
    print(f"\n--- Federated Round {rnd} ---")
    
    w_a, m_a, n_a = local_train(global_model, loader_a, epochs=1, device=device, noise_scale=0.01)
    w_b, m_b, n_b = local_train(global_model, loader_b, epochs=1, device=device, noise_scale=0.01)
    w_c, m_c, n_c = local_train(global_model, loader_c, epochs=1, device=device, noise_scale=0.01)

    # FedAvg with Secure Aggregation
    global_weights = fedavg([w_a, w_b, w_c], [m_a, m_b, m_c], [n_a, n_b, n_c])
    global_model.load_state_dict(global_weights)

    acc = evaluate(global_model, test_loader, device=device)
    print(f"Global Model Accuracy after round {rnd}: {acc:.2f}%")
