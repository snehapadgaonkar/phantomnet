"""
features.py
------------
Converts raw network/system telemetry logs into structured behavioral features.
These features serve as PhantomNet's "behavioral fingerprints."
"""

import pandas as pd
import numpy as np


def aggregate_flow_features(flow_df: pd.DataFrame, window_seconds: int = 60):
    """
    Aggregate raw network flow data into fixed-time-window features.

    Parameters:
        flow_df: DataFrame with columns
                 ['timestamp', 'src_ip', 'dst_ip', 'bytes', 'packets']
        window_seconds: aggregation window size
    """
    flow_df['timestamp'] = pd.to_datetime(flow_df['timestamp'])
    flow_df['window'] = (flow_df['timestamp'].astype('int64') // (window_seconds * 10**9)).astype(int)

    grouped = flow_df.groupby(['src_ip', 'window']).agg(
        feat_total_bytes=('bytes', 'sum'),
        feat_packet_count=('packets', 'sum'),
        feat_unique_dst_count=('dst_ip', 'nunique'),
        feat_avg_pkt_size=('bytes', lambda x: np.mean(x)),
    ).reset_index()

    grouped['feat_pkt_rate'] = grouped['feat_packet_count'] / window_seconds
    return grouped


def aggregate_syscall_features(sys_df: pd.DataFrame, window_seconds: int = 60):
    """
    Aggregates system call telemetry into behavioral features per process.
    """
    sys_df['timestamp'] = pd.to_datetime(sys_df['timestamp'])
    sys_df['window'] = (sys_df['timestamp'].astype('int64') // (window_seconds * 10**9)).astype(int)
    agg = sys_df.groupby(['pid', 'window']).agg(
        feat_syscall_count=('syscall', 'count'),
        feat_unique_syscalls=('syscall', 'nunique')
    ).reset_index()
    return agg
