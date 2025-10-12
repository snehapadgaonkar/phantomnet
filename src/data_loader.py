"""
data_loader.py
---------------
Handles dataset loading, preprocessing, scaling, and splitting for PhantomNet.

This module loads behavioral fingerprint CSVs, applies normalization, and
returns PyTorch DataLoaders for training, validation, and testing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml
import os


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


class FeatureDataset:
    """
    Converts a features CSV into numpy arrays (X, y)
    and applies StandardScaler normalization.
    """

    def __init__(self, csv_path: str, label_col: str = "label", scaler=None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        df = pd.read_csv(csv_path)

        if label_col in df.columns:
            self.y = df[label_col].values.astype(int)
            self.X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values
        else:
            self.y = None
            self.X = df.select_dtypes(include=[np.number]).values

        self.scaler = scaler or StandardScaler()

    def fit_transform(self):
        """Fit scaler and transform data."""
        self.X = self.scaler.fit_transform(self.X)
        return self.X, self.y

    def transform(self):
        """Transform using existing scaler."""
        self.X = self.scaler.transform(self.X)
        return self.X, self.y


def get_dataloaders(train_csv, val_csv=None, test_csv=None, batch_size=128):
    """Return PyTorch DataLoaders for train/val/test splits."""
    train = FeatureDataset(train_csv)
    X_train, y_train = train.fit_transform()

    if val_csv:
        val = FeatureDataset(val_csv, scaler=train.scaler)
        X_val, y_val = val.transform()
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

    if test_csv:
        test = FeatureDataset(test_csv, scaler=train.scaler)
        X_test, y_test = test.transform()
    else:
        X_test, y_test = None, None

    def to_loader(X, y):
        return DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.long)),
            batch_size=batch_size, shuffle=True
        )

    train_loader = to_loader(X_train, y_train)
    val_loader = to_loader(X_val, y_val)
    test_loader = to_loader(X_test, y_test) if X_test is not None else None

    return train_loader, val_loader, test_loader, train.scaler
