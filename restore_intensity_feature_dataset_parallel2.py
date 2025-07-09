import os
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from multiprocessing import Pool
from tqdm import tqdm

'''
Parallel intensity restoration using nearest-neighbor lookup:
- Reads decompressed PLYs, reads original BIN (xyz+intensity)
- For each decompressed point, finds nearest original point (KD-tree)
- Assigns intensity if within threshold, else error
- Outputs merged BIN files

Usage:
python restore_intensity_nn.py \
  --ply_root ./analysis/Q_8/decompressed \
  --orig_bin_root ./goose-pointcept/lidar \
  --out_bin_root ./analysis/Q_8/decompressed_with_intensity \
  --threshold 0.01 \
  --num_workers 8
  
  
  python restore_intensity_feature_dataset_parallel2.py \
  --ply_root goose-dataset/reno_decompressed_lidar/Q_512 \
  --orig_bin_root goose-dataset/lidar \
  --threshold 0.33 \
  --out_bin_root goose-dataset/reno_bin_decompressed_lidar/Q_512
  
'''

def read_ply_xyz(ply_path: Path):
    pcd = o3d.io.read_point_cloud(str(ply_path), format='ply')
    return np.asarray(pcd.points, dtype=np.float32)


def read_bin_xyz_intensity(bin_path: Path):
    data = np.fromfile(str(bin_path), dtype=np.float32)
    if data.size % 4 != 0:
        raise ValueError(f"Unexpected float count in {bin_path}: got {data.size}")
    pts = data.reshape(-1, 4)
    return pts[:, :3].astype(np.float32), pts[:, 3].astype(np.float32)


def convert_intensity_nn(args):
    ply_path, ply_root, orig_bin_root, out_bin_root, threshold = args
    rel = ply_path.relative_to(ply_root).with_suffix('.bin')
    orig_bin = orig_bin_root / rel
    out_bin = out_bin_root / rel
    if out_bin.exists():
        return out_bin
    
    out_bin.parent.mkdir(parents=True, exist_ok=True)

    # Load points
    xyz_dec = read_ply_xyz(ply_path)
    xyz_orig, intensity_orig = read_bin_xyz_intensity(orig_bin)

   # Build KD-tree on original xyz: wrap coords in a PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_orig)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # For each decompressed point, find NN
    recovered_i = np.empty((xyz_dec.shape[0],), dtype=np.float32)
    for idx, pt in enumerate(xyz_dec):
        try:
            [_, idxs, dists] = pcd_tree.search_knn_vector_3d(pt, 1)
        except RuntimeError:
            raise RuntimeError(f"KD-tree lookup failed for point index {idx} in {ply_path}")
         
        dist = np.sqrt(dists[0])
        if dist > threshold:
            raise ValueError(f"No original point within {threshold} for {ply_path} point index {idx} (dist={dist})")
        recovered_i[idx] = intensity_orig[idxs[0]]

    # Merge and write
    merged = np.hstack((xyz_dec, recovered_i.reshape(-1,1))).astype(np.float32)
    merged.tofile(str(out_bin))
    return out_bin


def main():
    parser = argparse.ArgumentParser(
        description='Restore intensity via nearest-neighbor matching'
    )
    parser.add_argument('--ply_root', '-p', type=str, required=True,
                        help='Root directory of decompressed PLYs')
    parser.add_argument('--orig_bin_root', '-b', type=str, required=True,
                        help='Root of original BINs (xyz+i)')
    parser.add_argument('--out_bin_root', '-o', type=str, required=True,
                        help='Output root for merged BINs')
    parser.add_argument('--threshold', '-t', type=float, default=0.01,
                        help='Distance threshold for NN matching')
    parser.add_argument('--num_workers', '-n', type=int, default=os.cpu_count(),
                        help='Parallel worker count')
    args = parser.parse_args()

    ply_root = Path(args.ply_root)
    orig_bin_root = Path(args.orig_bin_root)
    out_bin_root = Path(args.out_bin_root)
    threshold = args.threshold
    num_workers = args.num_workers

    # Gather PLY files
    ply_files = list(ply_root.rglob('*.ply'))
    print(f"Found {len(ply_files)} PLY files under {ply_root}")

    # Prepare tasks
    tasks = [(p, ply_root, orig_bin_root, out_bin_root, threshold) for p in ply_files]

    # Parallel processing
    with Pool(processes=num_workers) as pool:
        for out in tqdm(pool.imap(convert_intensity_nn, tasks), total=len(tasks), desc="Restoring intensity NN"):
            print(f"Restored: {out}")

    print("Intensity restoration (NN) complete.")

if __name__ == '__main__':
    main()
