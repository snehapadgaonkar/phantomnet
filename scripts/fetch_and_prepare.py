#!/usr/bin/env python3
"""
scripts/fetch_and_prepare.py

Download dataset (Kaggle or URL) and produce a small sample CSV
at data/processed/sample_features.csv suitable for PhantomNet.
"""

import os
import argparse
import subprocess
import tempfile
import shutil
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
import pyarrow


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

def download_kaggle(dataset_slug, dest_dir):
    """Download a Kaggle dataset using kaggle CLI. Requires ~/.kaggle/kaggle.json"""
    cmd = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", dest_dir, "--unzip"]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

def download_http(url, out_path):
    print(f"Downloading {url} -> {out_path}")
    urlretrieve(url, out_path)

def sample_and_write(input_path, out_csv, sample_n=2000, label_col_candidates=None):
    import pyarrow.parquet as pq

    print(f"Loading file: {input_path} (auto-detecting format)...")
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(input_path)
    elif ext == ".csv":
        df = pd.read_csv(input_path, low_memory=False, encoding_errors="ignore")
    elif ext == ".zip":
        # Handle zipped parquet/csv
        import zipfile, tempfile
        with zipfile.ZipFile(input_path, "r") as z:
            tmpdir = tempfile.mkdtemp()
            z.extractall(tmpdir)
            inner_files = [f for f in os.listdir(tmpdir) if f.endswith((".csv", ".parquet"))]
            if not inner_files:
                raise FileNotFoundError("No CSV or Parquet found inside ZIP")
            inner_path = os.path.join(tmpdir, inner_files[0])
            df = pd.read_parquet(inner_path) if inner_path.endswith(".parquet") else pd.read_csv(inner_path, low_memory=False)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} cols")
    num_df = df.select_dtypes(include=[np.number]).copy()

    # find label column
    if label_col_candidates is None:
        label_col_candidates = ["Label", "label", "attack", "Attack", "class", "flow_label"]
    label_col = next((c for c in label_col_candidates if c in df.columns), None)

    if label_col:
        labels = df[label_col].astype(str)
        labels = labels.apply(lambda s: 0 if s.lower() in ["benign", "normal", "normal."] else 1)
        num_df["label"] = labels.values
    else:
        num_df["label"] = 0

    # sample subset
    sampled = num_df.sample(n=min(sample_n, len(num_df)), random_state=42)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    sampled.to_csv(out_csv, index=False)
    print(f"✅ Saved {out_csv} ({len(sampled)} rows)")
    return out_csv

def find_csv_in_dir(d):
    found = []
    for root, _, files in os.walk(d):
        for f in files:
            if f.lower().endswith(".csv"):
                found.append(os.path.join(root, f))
    return found

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", type=str, help="kaggle dataset slug (user/dataset)")
    parser.add_argument("--url", type=str, help="direct CSV download URL")
    parser.add_argument("--local", type=str, help="path to a local CSV to process")
    parser.add_argument("--sample", type=int, default=2000, help="number of rows to sample for demo CSV")
    args = parser.parse_args()

    tmpdir = tempfile.mkdtemp(prefix="phantom_fetch_")
    try:
        if args.kaggle:
            print("Attempting Kaggle download for:", args.kaggle)
            download_kaggle(args.kaggle, tmpdir)
            found = find_csv_in_dir(tmpdir)
            if not found:
                raise FileNotFoundError("No CSV found inside Kaggle dataset files. Check dataset contents.")
            csv_path = found[0]
            out = sample_and_write(csv_path, os.path.join(OUT_DIR, "sample_features.csv"), sample_n=args.sample)
            return

        if args.url:
            out_path = os.path.join(tmpdir, "download.csv")
            download_http(args.url, out_path)
            out = sample_and_write(out_path, os.path.join(OUT_DIR, "sample_features.csv"), sample_n=args.sample)
            return

        if args.local:
            out = sample_and_write(args.local, os.path.join(OUT_DIR, "sample_features.csv"), sample_n=args.sample)
            return

        print("No download source provided. Example usage:")
        print(" python scripts/fetch_and_prepare.py --kaggle dhoogla/cicids2017 --sample 3000")
        print('Or: python scripts/fetch_and_prepare.py --url "https://path/to/flows.csv" --sample 2000')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
