import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import multiprocessing

"""
Parallel conversion of .bin LiDAR files to three separate .dat files (x, y, z) for LCP compression.
Strips out intensity and preserves directory structure.

Usage:
python bin2dat_parallel.py \
  --input_root ./goose-pointcept/lidar \
  --output_root ./goose-pointcept/dat_xyz_only \
  --num_workers 8
  
python bin2dat_parallel.py \
  --input_root /scratch/aniemcz/goose-pointcept/lidar \
  --output_root /scratch/aniemcz/goose-pointcept/dat_xyz_only_lidar
"""

def convert_file(args):
    bin_path, input_root, output_root = args
    # Compute relative path and output directory
    rel = bin_path.relative_to(input_root)
    stem = bin_path.stem
    out_dir = output_root / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read binary data as float32, reshape into N x 4
    data = np.fromfile(str(bin_path), dtype=np.float32)
    if data.size % 4 != 0:
        raise ValueError(f"Unexpected float count in {bin_path}: {data.size}")
    points = data.reshape(-1, 4)
    xyz = points[:, :3]

    # Write separate .dat files for x, y, z
    x_dat = out_dir / f"{stem}_x.dat"
    y_dat = out_dir / f"{stem}_y.dat"
    z_dat = out_dir / f"{stem}_z.dat"
    xyz[:, 0].astype(np.float32).reshape(1, -1).tofile(str(x_dat))
    xyz[:, 1].astype(np.float32).reshape(1, -1).tofile(str(y_dat))
    xyz[:, 2].astype(np.float32).reshape(1, -1).tofile(str(z_dat))

    return (x_dat, y_dat, z_dat)


def process_files(bin_files, input_root, output_root, num_workers):
    tasks = [(f, input_root, output_root) for f in bin_files]
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(convert_file, tasks),
            total=len(tasks),
            desc="Converting .bin to .dat files"
        ))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Strip intensity and write x,y,z-only .dat files in parallel"
    )
    parser.add_argument(
        '--input_root', '-i', type=str, required=True,
        help='Root directory of original .bin files'
    )
    parser.add_argument(
        '--output_root', '-o', type=str, required=True,
        help='Directory where x/y/z .dat files will be created'
    )
    parser.add_argument(
        '--num_workers', '-n', type=int, default=os.cpu_count(),
        help='Number of parallel workers (default: CPU count)'
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    # Gather all .bin files
    bin_files = list(input_root.rglob('*.bin'))
    print(f"Found {len(bin_files)} .bin files under {input_root}")

    # Process and create .dat files
    process_files(bin_files, input_root, output_root, args.num_workers)
    print(f"Completed conversion to .dat files at: {output_root}")
