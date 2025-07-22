#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

'''
HOW TO USE:
1) Generate your restored bins:
   python restore_intensity_feature_dataset_parallel2.py \
     --ply_root goose-dataset/ply_xyz_only_lidar \
     --orig_bin_root goose-dataset/lidar \
     --threshold 0.007 \
     --out_bin_root goose-dataset/test_restored_intensity_bin_lidar

2) Run the sanity check:
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

    orig_root     = Path(args.orig_bin_root)
    restored_root = Path(args.restored_bin_root)

    restored_files = sorted(restored_root.rglob("*.bin"))
    if not restored_files:
        print(f"No .bin files found under {restored_root}", file=sys.stderr)
        sys.exit(1)

    n_pass = 0
    n_fail = 0
    failures = []

    for rpath in tqdm(restored_files, desc="Checking"):
        rel       = rpath.relative_to(restored_root)
        orig_path = orig_root / rel
        if not orig_path.exists():
            failures.append(f"{rel}: missing original")
            n_fail += 1
            continue

        orig = read_bin(orig_path)
        rec  = read_bin(rpath)

        if orig.shape != rec.shape:
            failures.append(f"{rel}: shape mismatch orig {orig.shape} vs rec {rec.shape}")
            n_fail += 1
            continue

        # sort both by x,y,z
        order   = np.lexsort((orig[:,2], orig[:,1], orig[:,0]))
        orig_s  = orig[order]
        rec_s   = rec[order]
        xyz     = orig_s[:,:3]

        # find duplicates in original xyz
        # np.unique with return_counts to detect duplicate positions
        uniq_xyz, idx_first, counts = np.unique(
            xyz, axis=0, return_index=True, return_counts=True
        )
        # build a mask of positions that are NOT duplicates
        single_mask = np.zeros(len(xyz), dtype=bool)
        # mark each unique position that has count==1
        for pos_idx, ct in zip(idx_first, counts):
            if ct == 1:
                single_mask[pos_idx] = True

        # now compare only those single positions
        orig_i = orig_s[:,3][single_mask]
        rec_i  = rec_s[:,3][single_mask]

        if orig_i.size == 0:
            # nothing to check—all points were duplicates
            n_pass += 1
            continue

        if not np.array_equal(orig_i, rec_i):
            diffs = np.where(orig_i != rec_i)[0]
            msg = (f"{rel}: {len(diffs)} mismatches out of {orig_i.size} "
                   f"unique points; first at indices {diffs[:5].tolist()}")
            failures.append(msg)
            n_fail += 1
        else:
            n_pass += 1

    # summary
    print("\n==== Summary ====")
    total = len(restored_files)
    print(f"Checked {total} files: {n_pass} passed, {n_fail} FAILED.")
    if failures:
        print("\nFailures:")
        for msg in failures:
            print(" •", msg)
        sys.exit(2)
    else:
        print("All unique-point intensities perfectly restored!")
        sys.exit(0)


if __name__ == "__main__":
    main()
