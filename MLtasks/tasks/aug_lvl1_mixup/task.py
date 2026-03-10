"""
Mixup Training Augmentation Task
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Constants
BATCH_SIZE = 128
EPOCHS = 2
ALPHA = 0.2

def get_task_metadata():
    return {
        "id": "aug_lvl1_mixup",
        "protocol": "pytorch_task_v1"
    }

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataloaders(batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def build_model(device=None):
    if device is None:
        device = get_device()
        
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return model.to(device)

def mixup_data(x, y, alpha=1.0, device='cpu'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(model, train_loader, epochs=EPOCHS, device=None):
    if device is None:
        device = get_device()
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            inputs, targets_a, targets_b, lam = mixup_data(x, y, ALPHA, device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

def evaluate(model, val_loader, device=None):
    if device is None:
        device = get_device()
        
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
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

def save_artifacts(metrics, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

def main():
    print("=" * 60)
    print("Mixup Augmentation Task")
    print("=" * 60)
    
    device = get_device()
    set_seed()
    
    print("\nCreating dataloaders...")
    train_loader, val_loader = make_dataloaders()
    
    print("Building model...")
    model = build_model(device=device)
    
    print("Training model...")
    train(model, train_loader, device=device)
    
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device=device)
    
    print("\nSaving artifacts...")
    save_artifacts(val_metrics)
    
    print("\n" + "=" * 60)
    print("Quality Checks:")
    print("=" * 60)
    
    acc_pass = val_metrics['accuracy'] > 0.70
    print(f"{'✓' if acc_pass else '✗'} Val Accuracy > 0.70: {val_metrics['accuracy']:.4f}")
    
    all_pass = acc_pass
    return 0 if all_pass else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)