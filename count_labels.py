import os
import argparse
from pathlib import Path
import numpy as np
import csv
from multiprocessing import Pool
from functools import reduce
from tqdm import tqdm

"""
Count occurrences of each semantic class across all .label files.

Usage:
python count_labels.py \
  --label_root ./goose-pointcept/labels_mapped \
  --output_csv ./label_counts.csv
  
python count_labels.py \
  --label_root /scratch/aniemcz/goose-pointcept/labels_challenge \
  --output_csv ./label_counts.csv
"""

def count_file_labels(label_path: Path):
    """
    Read a .label file, extract semantic labels, and count per class.
    Returns a dict: {class_id: count, ...}
    """
    data = np.fromfile(str(label_path), dtype=np.uint32)
    sem = data & 0xFFFF
    # unique labels and their counts
    unique, counts = np.unique(sem, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def merge_counts(counts_list):
    """Merge a list of {class:count} dicts by summing counts."""
    merged = {}
    for d in counts_list:
        for k, v in d.items():
            merged[k] = merged.get(k, 0) + v
    return merged


def main():
    parser = argparse.ArgumentParser(description="Count semantic label occurrences.")
    parser.add_argument('--label_root', '-l', type=str, required=True,
                        help='Root directory of .label files')
    parser.add_argument('--output_csv', '-o', type=str, default=None,
                        help='Optional CSV output path')
    parser.add_argument('--num_workers', '-n', type=int, default=1,
                        help='Number of parallel workers')
    args = parser.parse_args()

    label_root = Path(args.label_root)
    label_files = list(label_root.rglob('*.label'))
    print(f"Found {len(label_files)} label files under {label_root}")

    # Count per file in parallel or serial
    if args.num_workers > 1:
        with Pool(processes=args.num_workers) as pool:
            counts_list = list(tqdm(pool.imap(count_file_labels, label_files),
                                    total=len(label_files), desc='Counting files'))
    else:
        counts_list = [count_file_labels(p) for p in tqdm(label_files, desc='Counting files')]

    # Merge all counts
    total_counts = merge_counts(counts_list)

    # Sort classes
    sorted_items = sorted(total_counts.items())  # list of (class_id, count)

    # Output
    if args.output_csv:
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['class_id', 'count'])
            for cls, cnt in sorted_items:
                writer.writerow([cls, cnt])
        print(f"Counts written to {args.output_csv}")
    else:
        # Print to console
        print("class_id,count")
        for cls, cnt in sorted_items:
            print(f"{cls},{cnt}")

if __name__ == '__main__':
    main()
