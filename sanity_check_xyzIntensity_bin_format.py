#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

"""
Sanity check script for verifying .bin files can be read and parsed
as (x, y, z, intensity) float32 point clouds.

Usage:
python sanity_check_bin_loading.py \
  --bin_root path/to/directory
"""

def check_bin_file(bin_path):
    try:
        scan = np.fromfile(str(bin_path), dtype=np.float32)
        if scan.size % 4 != 0:
            return False, f"Unexpected float count ({scan.size})"

        scan = scan.reshape((-1, 4))
        _ = scan[:, 0:3]  # xyz
        _ = scan[:, 3]    # remission
        return True, None

    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(
        description="Check if .bin files can be read and reshaped into (N, 4) float32 format"
    )
    parser.add_argument(
        "--bin_root", "-b", type=str, required=True,
        help="Directory to recursively scan for .bin files"
    )
    args = parser.parse_args()
    bin_root = Path(args.bin_root)

    bin_files = list(bin_root.rglob("*.bin"))
    if not bin_files:
        print(f"No .bin files found under {bin_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Checking {len(bin_files)} .bin files under {bin_root}...\n")

    n_ok = 0
    n_fail = 0
    failures = []

    for bin_path in tqdm(bin_files, desc="Validating"):
        ok, error = check_bin_file(bin_path)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            failures.append(f"{bin_path}: {error}")

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
