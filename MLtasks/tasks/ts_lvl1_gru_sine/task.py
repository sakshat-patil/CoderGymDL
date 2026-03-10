"""
GRU Time Series Forecasting Task
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Constants
SEQ_LENGTH = 20
HIDDEN_SIZE = 16
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.01

def get_task_metadata():
    return {
        "id": "ts_lvl1_gru_sine",
        "protocol": "pytorch_task_v1"
    }

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataloaders(seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE):
    x = np.linspace(0, 100, 1000)
    y = np.sin(x) + np.random.randn(1000) * 0.1
    
    X_data, y_data = [], []
    for i in range(len(y) - seq_length):
        X_data.append(y[i:i+seq_length])
        y_data.append(y[i+seq_length])
        
    X_tensor = torch.FloatTensor(np.array(X_data)).unsqueeze(-1)
    y_tensor = torch.FloatTensor(np.array(y_data)).unsqueeze(-1)
    
    split = int(0.8 * len(X_tensor))
    train_dataset = TensorDataset(X_tensor[:split], y_tensor[:split])
    val_dataset = TensorDataset(X_tensor[split:], y_tensor[split:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_tensor[:split], X_tensor[split:], y_tensor[:split], y_tensor[split:]

class GRUNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def build_model(device=None):
    if device is None:
        device = get_device()
    return GRUNet().to(device)

def train(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, device=None):
    if device is None:
        device = get_device()
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
    return train_losses, val_losses

def evaluate(model, data_loader, targets, device=None):
    if device is None:
        device = get_device()
        
    model.eval()
    all_outputs = []
    
    with torch.no_grad():
        for batch_X, _ in data_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            all_outputs.append(outputs.cpu())
            
    outputs_np = torch.cat(all_outputs).numpy()
    targets_np = targets.numpy()
    
    mse = mean_squared_error(targets_np, outputs_np)
    r2 = r2_score(targets_np, outputs_np)
    
    return {"mse": float(mse), "r2": float(r2), "predictions": outputs_np, "targets": targets_np}

def predict(model, input_tensor, device=None):
    if device is None:
        device = get_device()
    model.eval()
    with torch.no_grad():
        return model(input_tensor.to(device)).cpu().numpy()

def save_artifacts(train_losses, val_losses, val_metrics, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 4))
    plt.plot(val_metrics['targets'][:100], label='True')
    plt.plot(val_metrics['predictions'][:100], label='Predicted')
    plt.title("GRU Forecast (First 100 Val Steps)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "ts_forecast.png"))
    plt.close()
    
    metrics = {"val_mse": val_metrics["mse"], "val_r2": val_metrics["r2"]}
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

def main():
    print("=" * 60)
    print("GRU Time Series Forecasting Task")
    print("=" * 60)
    
    device = get_device()
    set_seed()
    
    print("\nCreating dataloaders...")
    train_loader, val_loader, _, _, _, y_val = make_dataloaders()
    
    print("Building model...")
    model = build_model(device=device)
    
    print("Training model...")
    train_losses, val_losses = train(model, train_loader, val_loader, device=device)
    
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, y_val, device=device)
    
    print("\nSaving artifacts...")
    save_artifacts(train_losses, val_losses, val_metrics)
    
    print("\n" + "=" * 60)
    print("Quality Checks:")
    print("=" * 60)
    
    r2_pass = val_metrics['r2'] > 0.50
    print(f"{'✓' if r2_pass else '✗'} Val R2 > 0.50: {val_metrics['r2']:.4f}")
    
    mse_pass = val_metrics['mse'] < 0.20
    print(f"{'✓' if mse_pass else '✗'} Val MSE < 0.20: {val_metrics['mse']:.4f}")
    
    all_pass = r2_pass and mse_pass
    return 0 if all_pass else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)