"""
AdamW and Cosine Annealing Optimization Task
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Constants
BATCH_SIZE = 64
EPOCHS = 2

def get_task_metadata():
    return {
        "id": "optim_lvl1_adamw_cosine",
        "protocol": "pytorch_task_v1"
    }

def set_seed(seed=42):
    torch.manual_seed(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataloaders(batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
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
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 10)
    )
    return model.to(device)

def train(model, train_loader, val_loader, epochs=EPOCHS, device=None):
    if device is None:
        device = get_device()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader))

    lrs = []
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

    return lrs, train_losses, val_losses

def evaluate(model, data_loader, device=None):
    if device is None:
        device = get_device()

    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct, total, eval_loss = 0, 0, 0.0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            eval_loss += criterion(outputs, y).item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return {
        "accuracy": correct / total,
        "loss": eval_loss / len(data_loader),
        "mse": 0.0,
        "r2": 0.0
    }

def predict(model, input_tensor, device=None):
    if device is None:
        device = get_device()
    model.eval()
    with torch.no_grad():
        return model(input_tensor.to(device))

def save_artifacts(lrs, train_losses, val_losses, train_metrics, val_metrics, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    # LR Schedule Plot
    plt.figure()
    plt.plot(lrs)
    plt.title("Cosine Annealing Warm Restarts Schedule")
    plt.xlabel("Batch Step")
    plt.ylabel("Learning Rate")
    plt.savefig(os.path.join(output_dir, "lr_schedule.png"))
    plt.close()

    # Missing Validation Loss Curve Plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', color='blue', marker='o')
    plt.plot(val_losses, label='Validation Loss', color='orange', marker='x')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    # Save both sets of metrics
    metrics = {
        "train": train_metrics,
        "val": val_metrics
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

def main():
    print("=" * 60)
    print("AdamW + Cosine Annealing Task")
    print("=" * 60)

    device = get_device()
    set_seed()

    print("\nCreating dataloaders...")
    train_loader, val_loader = make_dataloaders()

    print("Building model...")
    model = build_model(device=device)

    print("Training model...")
    lrs, train_losses, val_losses = train(model, train_loader, val_loader, device=device)

    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device=device)

    print("Evaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device=device)

    print("\nSaving artifacts...")
    save_artifacts(lrs, train_losses, val_losses, train_metrics, val_metrics)

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Train Accuracy: {train_metrics['accuracy']:.4f} | Train Loss: {train_metrics['loss']:.4f}")
    print(f"Val Accuracy:   {val_metrics['accuracy']:.4f} | Val Loss:   {val_metrics['loss']:.4f}")
    
    print("\n" + "=" * 60)
    print("Quality Checks:")
    print("=" * 60)
    
    acc_pass = val_metrics['accuracy'] > 0.40
    print(f"{'✓' if acc_pass else '✗'} Val Accuracy > 0.40: {val_metrics['accuracy']:.4f}")
    
    all_pass = acc_pass
    return 0 if all_pass else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)