"""
simulator.py
------------
Generates synthetic network traffic logs for testing the detection system.
Produces a CSV file with at least 5,000 entries covering:
  - Benign traffic
  - DoS (Denial of Service)
  - Brute Force
  - Port Scanning

Run this first to generate your dataset before training the classifier.
Usage:
    python simulator.py
"""

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

# ── Configuration ──────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
TOTAL_SAMPLES = 6000          # total log entries to generate
OUTPUT_DIR    = "data"
OUTPUT_FILE   = os.path.join(OUTPUT_DIR, "network_logs.csv")

# Class distribution (must sum to 1.0)
CLASS_DISTRIBUTION = {
    "BENIGN":       0.50,   # 3000 normal packets
    "DoS":          0.20,   # 1200 DoS attack packets
    "BruteForce":   0.15,   #  900 brute force packets
    "PortScan":     0.15,   #  900 port scan packets
}

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ── Feature generators per traffic type ────────────────────────────────────────

def generate_benign(n):
    """Normal web / email / DNS traffic patterns."""
    return pd.DataFrame({
        "duration":         np.random.uniform(0.01, 10.0, n),
        "packet_count":     np.random.randint(1, 50, n),
        "byte_count":       np.random.randint(40, 5000, n),
        "packets_per_sec":  np.random.uniform(0.1, 10.0, n),
        "bytes_per_sec":    np.random.uniform(40, 2000, n),
        "src_port":         np.random.choice([80, 443, 53, 25, 587, 993,
                                              np.random.randint(1024, 65535)], n),
        "dst_port":         np.random.choice([80, 443, 53, 25, 587, 993], n),
        "protocol":         np.random.choice(["TCP", "UDP", "ICMP"], n,
                                             p=[0.70, 0.25, 0.05]),
        "syn_flag_count":   np.random.randint(0, 3, n),
        "ack_flag_count":   np.random.randint(0, 5, n),
        "fin_flag_count":   np.random.randint(0, 2, n),
        "rst_flag_count":   np.zeros(n, dtype=int),
        "idle_time":        np.random.uniform(0.0, 5.0, n),
        "connection_count": np.random.randint(1, 10, n),
        "label":            ["BENIGN"] * n,
    })


def generate_dos(n):
    """DoS: floods with huge packet/byte volumes at high rate, mostly SYN."""
    return pd.DataFrame({
        "duration":         np.random.uniform(0.001, 0.5, n),
        "packet_count":     np.random.randint(500, 5000, n),
        "byte_count":       np.random.randint(50000, 500000, n),
        "packets_per_sec":  np.random.uniform(1000, 50000, n),
        "bytes_per_sec":    np.random.uniform(50000, 5000000, n),
        "src_port":         np.random.randint(1024, 65535, n),
        "dst_port":         np.random.choice([80, 443, 8080], n),
        "protocol":         np.random.choice(["TCP", "UDP", "ICMP"], n,
                                             p=[0.60, 0.30, 0.10]),
        "syn_flag_count":   np.random.randint(200, 2000, n),
        "ack_flag_count":   np.random.randint(0, 5, n),
        "fin_flag_count":   np.zeros(n, dtype=int),
        "rst_flag_count":   np.random.randint(0, 50, n),
        "idle_time":        np.zeros(n),
        "connection_count": np.random.randint(100, 1000, n),
        "label":            ["DoS"] * n,
    })


def generate_bruteforce(n):
    """Brute force: many repeated connection attempts to auth ports."""
    return pd.DataFrame({
        "duration":         np.random.uniform(0.1, 2.0, n),
        "packet_count":     np.random.randint(2, 10, n),
        "byte_count":       np.random.randint(80, 500, n),
        "packets_per_sec":  np.random.uniform(1.0, 20.0, n),
        "bytes_per_sec":    np.random.uniform(40, 500, n),
        "src_port":         np.random.randint(1024, 65535, n),
        "dst_port":         np.random.choice([22, 23, 3389, 21, 5900], n),
        "protocol":         ["TCP"] * n,
        "syn_flag_count":   np.random.randint(1, 4, n),
        "ack_flag_count":   np.random.randint(1, 4, n),
        "fin_flag_count":   np.random.randint(1, 3, n),
        "rst_flag_count":   np.random.randint(0, 3, n),
        "idle_time":        np.random.uniform(0.0, 0.5, n),
        "connection_count": np.random.randint(50, 500, n),
        "label":            ["BruteForce"] * n,
    })


def generate_portscan(n):
    """Port scan: short connections sweeping many different destination ports."""
    return pd.DataFrame({
        "duration":         np.random.uniform(0.0, 0.1, n),
        "packet_count":     np.random.randint(1, 3, n),
        "byte_count":       np.random.randint(40, 120, n),
        "packets_per_sec":  np.random.uniform(10, 500, n),
        "bytes_per_sec":    np.random.uniform(400, 5000, n),
        "src_port":         np.random.randint(1024, 65535, n),
        "dst_port":         np.random.randint(1, 65535, n),   # sweeping ports
        "protocol":         np.random.choice(["TCP", "UDP"], n, p=[0.80, 0.20]),
        "syn_flag_count":   np.random.randint(1, 3, n),
        "ack_flag_count":   np.zeros(n, dtype=int),
        "fin_flag_count":   np.zeros(n, dtype=int),
        "rst_flag_count":   np.random.randint(0, 2, n),
        "idle_time":        np.random.uniform(0.0, 0.01, n),
        "connection_count": np.random.randint(200, 2000, n),
        "label":            ["PortScan"] * n,
    })


# ── Timestamp generator ────────────────────────────────────────────────────────

def add_timestamps(df, base_time=None):
    """Add a realistic timestamp column to a dataframe."""
    if base_time is None:
        base_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [
        base_time + timedelta(seconds=i * random.uniform(0.01, 2.0))
        for i in range(len(df))
    ]
    df.insert(0, "timestamp", [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps])
    return df


def add_ip_addresses(df):
    """Add fake source and destination IP addresses."""
    src_ips = [f"192.168.{random.randint(1,10)}.{random.randint(1,254)}"
               for _ in range(len(df))]
    dst_ips = [f"10.0.{random.randint(0,5)}.{random.randint(1,50)}"
               for _ in range(len(df))]
    df.insert(1, "src_ip", src_ips)
    df.insert(2, "dst_ip", dst_ips)
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def generate_dataset(total=TOTAL_SAMPLES, output_path=OUTPUT_FILE):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frames = []
    for label, fraction in CLASS_DISTRIBUTION.items():
        n = int(total * fraction)
        if label == "BENIGN":
            frames.append(generate_benign(n))
        elif label == "DoS":
            frames.append(generate_dos(n))
        elif label == "BruteForce":
            frames.append(generate_bruteforce(n))
        elif label == "PortScan":
            frames.append(generate_portscan(n))

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)  # shuffle
    df = add_timestamps(df)
    df = add_ip_addresses(df)

    # Encode protocol as a numeric column (useful for ML later)
    protocol_map = {"TCP": 0, "UDP": 1, "ICMP": 2}
    df["protocol_num"] = df["protocol"].map(protocol_map).fillna(0).astype(int)

    df.to_csv(output_path, index=False)

    print(f"Dataset generated: {output_path}")
    print(f"Total records   : {len(df)}")
    print(f"\nClass distribution:")
    print(df["label"].value_counts().to_string())
    return df


if __name__ == "__main__":
    generate_dataset()
