"""
train.py
---------
Main training loop for PhantomNet.
Trains the model on behavioral features and saves checkpoints.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_loader import get_dataloaders, load_config
from model import PhantomNet


def train(config_path="config.yaml"):
    cfg = load_config(config_path)
    device = torch.device(cfg['training'].get('device', 'cpu'))

    # Load data
    train_loader, val_loader, _, scaler = get_dataloaders(
        cfg['data']['train_csv'],
        cfg['data'].get('val_csv'),
        cfg['data'].get('test_csv'),
        batch_size=cfg['training']['batch_size']
    )

    model = PhantomNet(**cfg['model']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])

    best_val_loss = float('inf')
    os.makedirs("artifacts/models", exist_ok=True)

    for epoch in range(cfg['training']['epochs']):
        model.train()
        total_loss = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                out = model(Xv)
                loss = criterion(out, yv)
                val_loss += loss.item()
                preds = out.argmax(dim=1)
                correct += (preds == yv).sum().item()
                total += yv.size(0)

        val_acc = correct / total
        val_loss /= len(val_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "scaler": scaler,
                "config": cfg
            }, "artifacts/models/phantomnet_best.pt")
            print("✅ Saved new best model")

    torch.save(model.state_dict(), "artifacts/models/phantomnet_final.pt")
    print("🎯 Training complete!")


if __name__ == "__main__":
    train()
