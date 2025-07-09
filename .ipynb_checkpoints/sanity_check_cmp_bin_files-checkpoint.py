#!/usr/bin/env python3
import sys
import argparse
import numpy as np
from pathlib import Path

def load_bin(path: Path):
    data = np.fromfile(path, dtype=np.float32)
    if data.size % 4 == 0:
        pts = data.reshape(-1, 4)
    else:
        # fallback: assume only xyz
        pts = data.reshape(-1, 3)
    return pts

def main():
    p = argparse.ArgumentParser(
        description="Check two .bin point‐clouds for exact shape & value equality"
    )
    p.add_argument("bin1", type=Path, help="First .bin file")
    p.add_argument("bin2", type=Path, help="Second .bin file")
    args = p.parse_args()

    a = load_bin(args.bin1)
    b = load_bin(args.bin2)

    if a.shape != b.shape:
        print(f"❌ Shape mismatch: {a.shape} vs {b.shape}")
        sys.exit(1)

    # exact comparison
    if not np.array_equal(a, b):
        diff = np.abs(a - b)
        max_diff = diff.max()
        idx = tuple(map(int, np.unravel_index(np.argmax(diff), a.shape)))
        print(f"❌ Data mismatch: max abs diff {max_diff} at index {idx}")
        sys.exit(1)

    print(f"✅ Files are identical: shape {a.shape}, all values match exactly.")

if __name__ == "__main__":
    main()
