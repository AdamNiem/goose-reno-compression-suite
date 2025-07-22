#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import multiprocessing

"""
Parallel conversion of separate x/y/z .dat files back into ASCII PLYs.

For each set of:
    .../foo_x.dat
    .../foo_y.dat
    .../foo_z.dat
this script will read the three float32 arrays, stack them to N×3,
and write:
    .../foo.ply
in ASCII PLY format (xyz only).

Usage:
python dat2ply_parallel.py \
  --input_root ./goose-pointcept/dat_xyz_only \
  --output_root ./goose-pointcept/ply_xyz_only \
  --num_workers 8
"""

def convert_dat_to_ply(args):
    x_dat, input_root, output_root = args

    # derive y/z paths and output path
    rel = x_dat.relative_to(input_root)
    stem = x_dat.stem[:-2]        # remove trailing "_x"
    y_dat = x_dat.with_name(stem + "_y.dat")
    z_dat = x_dat.with_name(stem + "_z.dat")

    # ensure y/z exist
    if not y_dat.exists() or not z_dat.exists():
        raise FileNotFoundError(f"Missing companion .dat for {x_dat}")

    # output .ply path
    rel_ply = rel.with_name(stem + ".ply")
    out_ply = output_root / rel_ply
    out_ply.parent.mkdir(parents=True, exist_ok=True)

    # load arrays
    x = np.fromfile(str(x_dat), dtype=np.float32)
    y = np.fromfile(str(y_dat), dtype=np.float32)
    z = np.fromfile(str(z_dat), dtype=np.float32)
    if not (x.size == y.size == z.size):
        raise ValueError(f"Length mismatch in {rel}")

    npts = x.size

    # write ASCII PLY
    with open(out_ply, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {npts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        # body
        for xi, yi, zi in zip(x, y, z):
            f.write(f"{xi} {yi} {zi}\n")

    return out_ply

def process_all(x_dat_files, input_root, output_root, num_workers):
    tasks = [(p, input_root, output_root) for p in x_dat_files]
    with multiprocessing.Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(convert_dat_to_ply, tasks),
                      total=len(tasks),
                      desc="Converting .dat → .ply"):
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Recombine x/y/z .dat into ASCII‐PLY in parallel"
    )
    parser.add_argument(
        '--input_root', '-i', type=str, required=True,
        help='Root directory of *_x.dat, *_y.dat, *_z.dat files'
    )
    parser.add_argument(
        '--output_root', '-o', type=str, required=True,
        help='Directory where .ply files will be created'
    )
    parser.add_argument(
        '--num_workers', '-n', type=int, default=os.cpu_count(),
        help='Number of parallel workers (default: CPU count)'
    )
    args = parser.parse_args()

    input_root  = Path(args.input_root)
    output_root = Path(args.output_root)

    # gather all *_x.dat (one task per triplet)
    x_dat_files = list(input_root.rglob("*_x.dat"))
    print(f"Found {len(x_dat_files)} x‐dat files under {input_root}")

    process_all(x_dat_files, input_root, output_root, args.num_workers)
    print(f"Done!  PLYs written to: {output_root}")
