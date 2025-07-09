import os
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from multiprocessing import Pool
from tqdm import tqdm

'''
Parallel intensity restoration:
- Reads PLYs, reads original BIN intensities, writes merged BINs
- Uses multiprocessing.Pool + tqdm for progress

Usage:
python restore_intensity_feature_dataset_parallel.py \
  --ply_root ./analysis/Q_8/decompressed \
  --orig_bin_root ./goose-pointcept/lidar \
  --out_bin_root ./analysis/Q_8/decompressed_with_intensity \
  --num_workers 8
  
python restore_intensity_feature_dataset_parallel.py \
  --ply_root ./analysis/Q_8/decompressed \
  --orig_bin_root ./goose-pointcept/lidar \
  --out_bin_root ./analysis/Q_8/decompressed_with_intensity \
  --num_workers 8
  
python restore_intensity_feature_dataset_parallel.py \
  --ply_root goose-dataset/reno_decompressed_lidar \
  --orig_bin_root goose-dataset/lidar \
  --out_bin_root goose-dataset/reno_bin_decompressed_lidar
'''

def read_ply_xyz(ply_path: Path):
    pcd = o3d.io.read_point_cloud(str(ply_path), format='ply')
    return np.asarray(pcd.points, dtype=np.float32)


def read_bin_intensity(bin_path: Path):
    data = np.fromfile(str(bin_path), dtype=np.float32)
    if data.size % 4 != 0:
        raise ValueError(f"Unexpected float count in {bin_path}: got {data.size}")
    points = data.reshape(-1, 4)
    return points[:, 3].astype(np.float32)


def convert_intensity(args):
    ply_path, orig_bin_root, out_bin_root = args
    # Relative path under ply_root is stored globally
    rel = ply_path.relative_to(PLY_ROOT).with_suffix('.bin')
    orig_bin = orig_bin_root / rel
    out_bin = out_bin_root / rel
    out_bin.parent.mkdir(parents=True, exist_ok=True)

    xyz = read_ply_xyz(ply_path)
    intensity = read_bin_intensity(orig_bin)
    if xyz.shape[0] != intensity.shape[0]:
        raise ValueError(
            f"Point count mismatch at {ply_path}: PLY {xyz.shape[0]} vs BIN {intensity.shape[0]}"
        )
    merged = np.hstack((xyz, intensity.reshape(-1,1)))
    merged.astype(np.float32).tofile(str(out_bin))
    return out_bin

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Restore intensity in parallel'  
    )
    parser.add_argument('--ply_root', '-p', type=str, required=True,
                        help='Root directory of decompressed PLYs')
    parser.add_argument('--orig_bin_root', '-b', type=str, required=True,
                        help='Root of original BINs')
    parser.add_argument('--out_bin_root', '-o', type=str, required=True,
                        help='Output root for merged BINs')
    parser.add_argument('--num_workers', '-n', type=int, default=os.cpu_count(),
                        help='Parallel worker count')
    args = parser.parse_args()

    PLY_ROOT = Path(args.ply_root)
    orig_bin_root = Path(args.orig_bin_root)
    out_bin_root = Path(args.out_bin_root)
    num_workers = args.num_workers

    # Gather files and process
    ply_files = list(PLY_ROOT.rglob('*.ply'))
    print(f"Found {len(ply_files)} PLY files under {PLY_ROOT}")
    tasks = [(p, orig_bin_root, out_bin_root) for p in ply_files]

    with Pool(processes=num_workers) as pool:
        for out in tqdm(pool.imap(convert_intensity, tasks), total=len(tasks), desc="Restoring intensity"):
            print(f"Restored: {out}")

    print("Intensity restoration complete.")
