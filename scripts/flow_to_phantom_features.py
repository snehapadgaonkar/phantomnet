#!/usr/bin/env python3
"""
scripts/flow_to_phantom_features.py

Fallback converter that picks common numeric columns from a flow CSV and writes
data/processed/sample_features.csv with `label` column (if present).
"""

import pandas as pd
import os
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT = os.path.join(ROOT, "data", "processed", "sample_features.csv")
os.makedirs(os.path.dirname(DEFAULT_OUT), exist_ok=True)

def convert(infile, out_path=DEFAULT_OUT, keep_cols=None):
    print(f"📂 Loading file: {infile}")
    df = pd.read_csv(infile, low_memory=False)

    if keep_cols:
        cols = [c for c in keep_cols if c in df.columns]
    else:
        # choose some typical flow columns if present
        candidates = [
            "SrcIP", "DstIP", "FlowDuration", "TotLenFwdPkts", "TotLenBwdPkts",
            "FwdPktLenAvg", "BwdPktLenAvg", "FlowBytes/s", "FlowPackets/s", "Label", "label"
        ]
        cols = [c for c in candidates if c in df.columns]
        if not cols:
            cols = df.select_dtypes(include=["number"]).columns.tolist()[:20]

    # Extract selected columns
    out_df = df[cols].copy()

    # Ensure label column exists
    if "Label" in out_df.columns:
        out_df = out_df.rename(columns={"Label": "label"})
    if "label" not in out_df.columns:
        out_df["label"] = 0

    # Save processed features
    print(f"💾 Saving processed file to: {out_path}")
    out_df.to_csv(out_path, index=False)
    print(f"✅ Wrote processed features: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="path to raw flow CSV")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    convert(args.infile, args.out)
