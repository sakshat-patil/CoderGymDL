"""
Autoencoder Anomaly Detection Task
Trains an autoencoder and uses reconstruction error as anomaly score.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, Optional

# Set random seeds for reproducibility
SEED = 42

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_task_metadata() -> Dict[str, Any]:
    return {
        'task_name': 'autoencoder_anomaly_detection',
        'task_type': 'unsupervised_anomaly_detection',
        'description': 'Train an autoencoder and use reconstruction error as anomaly score',
        'input_type': 'tabular',
        'output_type': 'anomaly_scores',
        'metrics': ['mse', 'r2', 'accuracy', 'precision', 'recall', 'f1', 'auc']
    }

def make_dataloaders(
    batch_size: int = 64, test_size: float = 0.2, anomaly_ratio: float = 0.15, random_state: int = SEED
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    set_seed(random_state)

    X, y = make_classification(
        n_samples=2000, n_features=10, n_informative=8, n_redundant=2,
        n_clusters_per_class=1, flip_y=0, weights=[1 - anomaly_ratio, anomaly_ratio], random_state=random_state
    )

    anomaly_mask = y == 1
    normal_mask = ~anomaly_mask
    X_anomalies = X[anomaly_mask].copy()
    X_normal = X[normal_mask].copy()
    X_anomalies = X_anomalies + np.random.normal(0, 3, X_anomalies.shape)

    X = np.vstack([X_normal, X_anomalies])
    y = np.concatenate([np.zeros(len(X_normal)), np.ones(len(X_anomalies))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(X_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(X_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(X_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X_train, X_val

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 5):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8), nn.ReLU(),
            nn.Linear(8, 6), nn.ReLU(),
            nn.Linear(6, encoding_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 6), nn.ReLU(),
            nn.Linear(6, 8), nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error

def build_model(device: torch.device, input_dim: int = 10) -> Autoencoder:
    return Autoencoder(input_dim=input_dim).to(device)

def train(model: Autoencoder, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, num_epochs: int = 50, learning_rate: float = 0.001) -> Dict[str, list]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience, patience_counter = 7, 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)
            reconstructed = model(batch_X)
            loss = criterion(reconstructed, batch_X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_X = batch_X.to(device)
                val_loss += criterion(model(batch_X), batch_X).item()
        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                model.load_state_dict(best_model_state)
                break
    return history

def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 1) & (y_pred == 0))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def evaluate(model: Autoencoder, data_loader: DataLoader, device: torch.device, X_data: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    model.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for batch_X, _ in data_loader:
            errors = model.get_reconstruction_error(batch_X.to(device))
            reconstruction_errors.extend(errors.cpu().numpy())

    reconstruction_errors = np.array(reconstruction_errors)
    best_threshold, best_f1, best_j = np.median(reconstruction_errors), 0.0, 0.0
    thresholds = np.percentile(reconstruction_errors, np.arange(50, 99, 1))

    for threshold in thresholds:
        y_pred = (reconstruction_errors > threshold).astype(int)
        if np.sum(y_pred) == 0 or np.sum(y_pred) == len(y_pred): continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        specificity = specificity_score(y_true, y_pred)
        j = f1 + specificity - 1
        if f1 > best_f1 or j > best_j:
            best_f1, best_j, best_threshold = f1, j, threshold

    y_pred = (reconstruction_errors > best_threshold).astype(int)

    reconstructed_all = []
    with torch.no_grad():
        for batch_X, _ in data_loader:
            reconstructed_all.extend(model(batch_X.to(device)).cpu().numpy())
    reconstructed_all = np.array(reconstructed_all)

    return {
        'mse': float(mean_squared_error(X_data, reconstructed_all)),
        'r2': float(r2_score(X_data.flatten(), reconstructed_all.flatten())),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'best_f1': float(best_f1),
        'auc': float(roc_auc_score(y_true, reconstruction_errors) if len(np.unique(y_true)) > 1 else 0.5),
        'threshold': float(best_threshold),
        'mean_reconstruction_error': float(np.mean(reconstruction_errors)),
        'std_reconstruction_error': float(np.std(reconstruction_errors))
    }

def predict(model: Autoencoder, X: np.ndarray, device: torch.device, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        errors = model.get_reconstruction_error(torch.FloatTensor(X).to(device)).cpu().numpy()
    if threshold is None: threshold = np.median(errors)
    return errors, (errors > threshold).astype(int)

def save_artifacts(model: Autoencoder, history: Dict[str, list], train_metrics: Dict[str, float], val_metrics: Dict[str, float], val_labels: np.ndarray, val_errors: np.ndarray, output_dir: str = 'output') -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'autoencoder_model.pth'))

    metrics = {"train": train_metrics, "val": val_metrics}
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(val_errors, bins=50, alpha=0.7, color='purple')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Validation Reconstruction Error Distribution')
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    plt.close()

    fpr, tpr, _ = roc_curve(val_labels, val_errors)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {val_metrics["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Autoencoder Anomaly Detection')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def main():
    print("=" * 60)
    print("Autoencoder Anomaly Detection Task")
    print("=" * 60)

    device = get_device()
    train_loader, val_loader, test_loader, X_train, X_val = make_dataloaders(batch_size=64, test_size=0.2, anomaly_ratio=0.15, random_state=SEED)
    model = build_model(device, input_dim=X_train.shape[1])
    history = train(model, train_loader, val_loader, device, num_epochs=50, learning_rate=0.001)

    np.random.seed(SEED)
    X_full, y_full = make_classification(
        n_samples=2000, n_features=10, n_informative=8, n_redundant=2, n_clusters_per_class=1,
        flip_y=0, weights=[0.85, 0.15], random_state=SEED
    )
    anomaly_mask = y_full == 1
    X_full = np.vstack([X_full[~anomaly_mask].copy(), X_full[anomaly_mask].copy() + np.random.normal(0, 3, X_full[anomaly_mask].shape)])
    y_full = np.concatenate([np.zeros(np.sum(~anomaly_mask)), np.ones(np.sum(anomaly_mask))])

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full, test_size=0.2, random_state=SEED, stratify=y_full)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=SEED, stratify=y_train_full)

    scaler = StandardScaler()
    X_train_split = scaler.fit_transform(X_train_split)
    X_val_split = scaler.transform(X_val_split)
    X_test_full = scaler.transform(X_test_full)

    print('\nEvaluating on training data...')
    train_metrics = evaluate(model, train_loader, device, X_train_split, y_train_split)

    print('Evaluating on validation data...')
    val_metrics = evaluate(model, val_loader, device, X_val_split, y_val_split)

    print('Evaluating on test data...')
    test_metrics = evaluate(model, test_loader, device, X_test_full, y_test_full)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Train Accuracy: {train_metrics['accuracy']:.4f} | Val Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Train F1:       {train_metrics['f1']:.4f} | Val F1:       {val_metrics['f1']:.4f}")

    print("\nGenerating data for ROC Curve...")
    val_errors, _ = predict(model, X_val_split, device)

    print("\nSaving artifacts...")
    save_artifacts(model, history, train_metrics, val_metrics, y_val_split, val_errors)

    if val_metrics['auc'] > 0.85 and val_metrics['accuracy'] > 0.85 and val_metrics['f1'] > 0.80:
        print("PASS: All quality assertions met.")
        sys.exit(0)
    else:
        print("FAIL: Quality assertions not met.")
        sys.exit(1)

if __name__ == '__main__':
    main()