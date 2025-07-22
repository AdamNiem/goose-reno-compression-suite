#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

'''
HOW TO USE:
Will need to first create the dataset to test that it works:
python restore_intensity_feature_dataset_parallel2.py   --ply_root goose-dataset/ply_xyz_only_lidar   --orig_bin_root goose-dataset/lidar   --threshold 0.007   --out_bin_root goose-dataset/test_restored_intensity_bin_lidar

And then to run:
python3 sanity_check_restored_intensity.py \
  --orig_bin_root goose-dataset/lidar \
  --restored_bin_root goose-dataset/test_restored_intensity_bin_lidar
'''

def read_bin(path):
    """Read a .bin of float32s and reshape (-1,4) -> (x,y,z,i)."""
    data = np.fromfile(str(path), dtype=np.float32)
    if data.size % 4 != 0:
        raise ValueError(f"Unexpected element count in {path}: {data.size}")
    return data.reshape(-1, 4)

def main():
    parser = argparse.ArgumentParser(
        description="Sanity‐check restored intensities against original BINs"
    )
    parser.add_argument(
        "--orig_bin_root", "-b", required=True,
        help="Root directory of original .bin files (x,y,z,intensity)"
    )
    parser.add_argument(
        "--restored_bin_root", "-r", required=True,
        help="Root directory of your restored .bin files"
    )
    args = parser.parse_args()

    orig_root = Path(args.orig_bin_root)
    restored_root = Path(args.restored_bin_root)

    # gather all restored bins
    restored_files = sorted(restored_root.rglob("*.bin"))
    if not restored_files:
        print(f"No .bin files found under {restored_root}", file=sys.stderr)
        sys.exit(1)

    n_pass = 0
    n_fail = 0
    failures = []

    for rpath in tqdm(restored_files, desc="Checking"):
        # derive the corresponding original path
        rel = rpath.relative_to(restored_root)
        orig_path = orig_root / rel
        if not orig_path.exists():
            print(f"[WARN] Original file missing: {orig_path}", file=sys.stderr)
            n_fail += 1
            failures.append(str(rpath)+" -> missing orig")
            continue

        orig = read_bin(orig_path)
        rec  = read_bin(rpath)

        if orig.shape != rec.shape:
            failures.append(f"{rel}: shape mismatch orig {orig.shape} vs rec {rec.shape}")
            n_fail += 1
            continue

        # sort both arrays by xyz to account for reordering
        def sort_by_xyz(a):
            return a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]

        orig_sorted = sort_by_xyz(orig)
        rec_sorted = sort_by_xyz(rec)

        if not np.array_equal(orig_sorted[:, 3], rec_sorted[:, 3]):
            diffs = np.where(orig_sorted[:, 3] != rec_sorted[:, 3])[0]
            # Report first few
            diffs_short = diffs[:5]
            msg = (f"{rel}: {len(diffs)} mismatched intensities; "
                   f"first mismatches at indices {diffs_short.tolist()}")
            failures.append(msg)
            n_fail += 1
        else:
            n_pass += 1

    print("\n==== Summary ====")
    print(f"Checked {len(restored_files)} files: {n_pass} ok, {n_fail} FAILED.")
    if failures:
        print("\nFailures:")
        for msg in failures:
            print(" •", msg)
        sys.exit(2)
    else:
        print("All intensities perfectly restored!")
        sys.exit(0)

if __name__ == "__main__":
    main()
