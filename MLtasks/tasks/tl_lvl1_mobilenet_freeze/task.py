"""
MobileNetV2 Feature Extraction Transfer Learning Task
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Constants
BATCH_SIZE = 32
EPOCHS = 1
SUBSET_SIZE = 2000

def get_task_metadata():
    return {
        "id": "tl_lvl1_mobilenet_freeze",
        "protocol": "pytorch_task_v1"
    }

def set_seed(seed=42):
    torch.manual_seed(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataloaders(batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    dataset = torch.utils.data.Subset(dataset, range(SUBSET_SIZE))
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def build_model(device=None):
    if device is None:
        device = get_device()
        
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace head
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    return model.to(device)

def train(model, train_loader, epochs=EPOCHS, device=None):
    if device is None:
        device = get_device()
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

def evaluate(model, data_loader, device=None):
    if device is None:
        device = get_device()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return {
        "accuracy": correct / total,
        "mse": 0.0,
        "r2": 0.0
    }

def predict(model, input_tensor, device=None):
    if device is None:
        device = get_device()
    model.eval()
    with torch.no_grad():
        return model(input_tensor.to(device))

def save_artifacts(train_metrics, val_metrics, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    metrics = {
        "train": train_metrics,
        "val": val_metrics
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

def main():
    print("=" * 60)
    print("MobileNetV2 Transfer Learning Task")
    print("=" * 60)

    device = get_device()
    set_seed()

    print("\nCreating dataloaders...")
    train_loader, val_loader = make_dataloaders()

    print("Building model...")
    model = build_model(device=device)

    print("Training model...")
    train(model, train_loader, device=device)

    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device=device)

    print("Evaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device=device)

    print("\nSaving artifacts...")
    save_artifacts(train_metrics, val_metrics)

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Val Accuracy:   {val_metrics['accuracy']:.4f}")
    
    print("\n" + "=" * 60)
    print("Quality Checks:")
    print("=" * 60)
    
    acc_pass = val_metrics['accuracy'] > 0.60
    print(f"{'✓' if acc_pass else '✗'} Val Accuracy > 0.60: {val_metrics['accuracy']:.4f}")
    
    all_pass = acc_pass
    return 0 if all_pass else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)