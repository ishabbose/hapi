"""Training loops for the 3D autoencoder baseline."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from hapi.models import SimpleBaseline3DAutoencoder


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_split(model, data_loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs, batch)
            total_loss += loss.item() * batch.size(0)
    return total_loss / len(data_loader.dataset)


def train_autoencoder(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-3,
    seed: int = 42,
    checkpoint_path=None,
) -> dict[str, float]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = SimpleBaseline3DAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_train = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            total_train += loss.item() * batch.size(0)
        train_mse = total_train / len(train_loader.dataset)
        val_mse = evaluate_split(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1:02d}/{epochs} | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")

    metrics = {
        "train_mse": evaluate_split(model, train_loader, criterion, device),
        "val_mse": evaluate_split(model, val_loader, criterion, device),
        "test_mse": evaluate_split(model, test_loader, criterion, device),
    }
    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved weights: {checkpoint_path}")
    return metrics
