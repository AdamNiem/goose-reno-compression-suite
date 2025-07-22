#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

"""
Sanity check script for verifying .label files can be read and parsed
as uint32 vectors with semantic and instance label extraction.

Semantic labels must be integers in [0, 8].

Usage:
python sanity_check_label_loading.py \
  --label_root path/to/label_directory
"""

def check_label_file(label_path):
    try:
        label = np.fromfile(str(label_path), dtype=np.uint32)
        label = label.reshape((-1,))  # Ensure 1D

        # Extract semantic and instance labels
        sem_label = label & 0xFFFF
        inst_label = label >> 16

        # Semantic labels must be between 0 and 8 inclusive
        if sem_label.min() < 0 or sem_label.max() > 8:
            return False, f"Semantic labels out of bounds: min={sem_label.min()}, max={sem_label.max()}"

        return True, None

    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(
        description="Check if .label files can be loaded and have semantic labels in [0, 8]"
    )
    parser.add_argument(
        "--label_root", "-l", type=str, required=True,
        help="Directory to recursively scan for .label files"
    )
    args = parser.parse_args()
    label_root = Path(args.label_root)

    label_files = list(label_root.rglob("*.label"))
    if not label_files:
        print(f"No .label files found under {label_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Checking {len(label_files)} .label files under {label_root}...\n")

    n_ok = 0
    n_fail = 0
    failures = []

    for label_path in tqdm(label_files, desc="Validating"):
        ok, error = check_label_file(label_path)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            failures.append(f"{label_path}: {error}")

    print("\n==== Summary ====")
    print(f"Passed: {n_ok}")
    print(f"Failed: {n_fail}")
    if failures:
        print("\nFailures:")
        for msg in failures:
            print(" •", msg)

    sys.exit(1 if n_fail > 0 else 0)

if __name__ == "__main__":
    main()
