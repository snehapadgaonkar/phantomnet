"""
utils.py
---------
Helper utilities for PhantomNet (directory setup, saving JSON, etc.)
"""

import os
import json


def ensure_dirs():
    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs("artifacts/logs", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)
