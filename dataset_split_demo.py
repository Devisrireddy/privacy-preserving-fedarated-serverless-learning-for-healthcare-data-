# dataset_split_demo.py
from torchvision import datasets, transforms
from torch.utils.data import random_split

# Transform for preprocessing
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load dataset
train_data = datasets.ImageFolder(r"E:\archive\chest_xray\chest_xray\train", transform=transform)

# Split dataset into 3 clients
total_size = len(train_data)
client_a_size = int(0.3 * total_size)
client_b_size = int(0.4 * total_size)
client_c_size = total_size - client_a_size - client_b_size

client_a, client_b, client_c = random_split(train_data, [client_a_size, client_b_size, client_c_size])

# Show results
print(f"Total dataset size: {total_size}")
print(f"Hospital A size: {len(client_a)}")
print(f"Hospital B size: {len(client_b)}")
print(f"Hospital C size: {len(client_c)}")
