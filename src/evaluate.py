"""
evaluate.py
------------
Evaluates PhantomNet model on the test or validation dataset and prints metrics.
"""

import torch
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import get_dataloaders, load_config
from model import PhantomNet


def evaluate(config_path="config.yaml", model_path="artifacts/models/phantomnet_best.pt"):
    cfg = load_config(config_path)
    device = torch.device(cfg['training'].get('device', 'cpu'))
    _, val_loader, test_loader, scaler = get_dataloaders(
        cfg['data']['train_csv'],
        cfg['data'].get('val_csv'),
        cfg['data'].get('test_csv'),
        batch_size=cfg['training']['batch_size']
    )

    model = PhantomNet(**cfg["model"]).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    loader = test_loader or val_loader

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            out = model(X)
            preds = out.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    evaluate()
