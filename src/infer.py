"""
infer.py
---------
Performs batch inference using the trained PhantomNet model.
"""

import argparse
import torch
import pandas as pd
import numpy as np
from model import PhantomNet
from data_loader import load_config
from sklearn.preprocessing import StandardScaler

def load_model(model_path: str, cfg: dict):
    device = torch.device(cfg['training'].get('device', 'cpu'))
    
    # Allow safe loading of objects like StandardScaler
    torch.serialization.add_safe_globals([StandardScaler, np.ndarray, np.float64, np.float32, np.int64, np.int32])
    
    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    
    model = PhantomNet(**cfg['model']).to(device)
    model.load_state_dict(ckpt['model_state'])
    scaler = ckpt.get('scaler')
    model.eval()
    return model, scaler, device


def infer_from_dataframe(df: pd.DataFrame, model, scaler, device):
    X = df.select_dtypes(include=[np.number]).values
    X_scaled = scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(X_t)
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = (probs > 0.5).astype(int)

    df_out = df.copy()
    df_out["prediction"] = preds
    df_out["prob_attack"] = probs
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    parser.add_argument("--model", type=str, default="artifacts/models/phantomnet_best.pt", help="Path to model checkpoint")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    model, scaler, device = load_model(args.model, cfg)
    
    df = pd.read_csv(args.input)
    results = infer_from_dataframe(df, model, scaler, device)
    
    results.to_csv(args.output, index=False)
    print(f"✅ Predictions saved to {args.output}")
