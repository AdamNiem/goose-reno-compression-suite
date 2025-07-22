#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from multiprocessing import Pool
from tqdm import tqdm

'''
Parallel dequantization + intensity restoration:
- Reads your ASCII PLYs (quantized xyz-only)
- Reverses the 18bit→float quantization
- Reads original BIN (xyz+intensity)
- Builds a KD-tree on original xyz
- For each dequantized point, does a 1‑NN lookup
- Writes out merged BIN (x,y,z,i)

Usage:
python dequant_and_restore_intensity.py \
  --ply_root    ./quantized_ply_xyz_only_lidar \
  --orig_bin_root ./goose-pointcept/lidar \
  --out_bin_root  ./quantized_with_intensity \
  --threshold 0.01 \
  --num_workers 8
'''

def read_ply_xyz(ply_path: Path):
    """Load an ASCII PLY with only x y z as floats via Open3D."""
    pcd = o3d.io.read_point_cloud(str(ply_path), format='ply')
    return np.asarray(pcd.points, dtype=np.float32)

def reverse_quantize(coords_q: np.ndarray) -> np.ndarray:
    """
    Reverse 18‑bit (1 mm) quantization:
      coords_q = round(orig/0.001) + 131072
    ⇒ orig = (coords_q - 131072) * 0.001
    """
    return (coords_q - 131072.0) * 0.001

def read_bin_xyz_intensity(bin_path: Path):
    """Read float32 .bin and split into (N×3 xyz, N intensities)."""
    data = np.fromfile(str(bin_path), dtype=np.float32)
    if data.size % 4 != 0:
        raise ValueError(f"Unexpected float count in {bin_path}: {data.size}")
    pts = data.reshape(-1, 4)
    return pts[:, :3].astype(np.float32), pts[:, 3].astype(np.float32)

def convert_intensity_nn(args):
    """
    Worker: dequantize + restore intensity for a single PLY.
    Returns the path to the written BIN.
    """
    ply_path, ply_root, orig_bin_root, out_bin_root, threshold, no_threshold = args

    # derive relative path → original bin & output bin
    rel     = ply_path.relative_to(ply_root)
    orig_bin = orig_bin_root / rel.with_suffix('.bin')
    out_bin  = out_bin_root  / rel.with_suffix('.bin')
    out_bin.parent.mkdir(parents=True, exist_ok=True)

    # 1) load & dequantize
    xyz_dec_q = read_ply_xyz(ply_path)
    xyz_dec   = reverse_quantize(xyz_dec_q)

    # 2) load original xyz+intensity
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
         
        if not no_threshold:
            dist = np.sqrt(dists[0])
            if dist > threshold:
                raise ValueError(f"No original point within {threshold} for {ply_path} point index {idx} (dist={dist})")
        recovered_i[idx] = intensity_orig[idxs[0]]

    # Merge and write
    merged = np.hstack((xyz_dec, recovered_i.reshape(-1,1))).astype(np.float32)
    merged.tofile(str(out_bin))
    return out_bin


def main():
    p = argparse.ArgumentParser(
        description="Dequantize ASCII‐PLY → restore intensities → write BIN"
    )
    p.add_argument("--ply_root",      "-p", required=True,
                   help="Root of your quantized-ascii PLYs (xyz-only)")
    p.add_argument("--orig_bin_root", "-b", required=True,
                   help="Root of original BINs (xyz+intensity)")
    p.add_argument("--out_bin_root",  "-o", required=True,
                   help="Where to write merged BINs")
    p.add_argument("--threshold",     "-t", type=float, default=0.01,
                   help="Max NN distance [m]")
    p.add_argument("--no_threshold",  "-s", action="store_true",
                   help="Disable distance‐threshold check")
    p.add_argument("--num_workers",   "-n", type=int,
                   default=os.cpu_count(),
                   help="Number of parallel workers")
    args = p.parse_args()

    ply_root      = Path(args.ply_root)
    orig_bin_root = Path(args.orig_bin_root)
    out_bin_root  = Path(args.out_bin_root)
    threshold     = args.threshold
    no_threshold  = args.no_threshold
    num_workers   = args.num_workers

    # collect all PLYs
    ply_files = list(ply_root.rglob("*.ply"))
    if not ply_files:
        raise RuntimeError(f"No PLYs found under {ply_root}")

    tasks = [
        (ply, ply_root, orig_bin_root, out_bin_root, threshold, no_threshold)
        for ply in ply_files
    ]

    with Pool(processes=num_workers) as pool:
        for out in tqdm(pool.imap_unordered(convert_intensity_nn, tasks),
                        total=len(tasks),
                        desc="Dequantize+Restore"):
            print(f"Wrote: {out}")

    print("All done!")

if __name__ == "__main__":
    main()
