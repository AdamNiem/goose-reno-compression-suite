import os
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from multiprocessing import Pool
from tqdm import tqdm

"""
Parallel label restoration via nearest-neighbor matching:
- Reads decompressed BINs for geometry (xyz)
- Reads original BINs for geometry+intensity, but here we only use xyz
- Reads original LABEL files to get semantic label per point
- For each decompressed point, finds nearest original point (KD-tree)
- Assigns its semantic label if within threshold, else error
- Writes a new .label file (uint32 semantics same bit layout) preserving directory structure

Usage:
python create_labels_4_decompressed_lidar.py \
  --decomp_bin_root ./analysis/Q_8/decompressed \
  --orig_bin_root ./goose-pointcept/lidar \
  --orig_label_root ./goose-pointcept/labels_challenge \
  --out_label_root ./analysis/Q_8/labels_restored \
  --threshold 0.01 \
  --num_workers 8

Q_lvl="Q_8"
python create_labels_4_decompressed_lidar.py \
  --decomp_bin_root /scratch/aniemcz/goose-pointcept/reno_bin_decompressed_lidar/${Q_lvl} \
  --orig_bin_root /scratch/aniemcz/goose-pointcept/lidar \
  --orig_label_root /scratch/aniemcz/goose-pointcept/labels_challenge \
  --out_label_root /scratch/aniemcz/goose-pointcept/reno_bin_decompressed_labels_challenge/${Q_lvl} \
  --threshold 0.007
  
"""

def read_bin_xyz(bin_path: Path):
    data = np.fromfile(str(bin_path), dtype=np.float32)
    if data.size % 4 != 0:
        raise ValueError(f"Unexpected float count in {bin_path}: {data.size}")
    pts = data.reshape(-1, 4)
    return pts[:, :3].astype(np.float32)


def read_label(label_path: Path):
    label = np.fromfile(str(label_path), dtype=np.uint32)
    sem = label & 0xFFFF
    inst = label >> 16
    return sem.astype(np.uint32), inst.astype(np.uint32)


def convert_labels_nn(args):
    decomp_bin_path, decomp_bin_root, orig_bin_root, orig_label_root, out_label_root, threshold = args
    rel = decomp_bin_path.relative_to(decomp_bin_root)
    
    # Replace `_vls128` with `_goose` in the filename (keep the directory structure)
    label_name = rel.stem.replace('_vls128', '_goose').replace('_pcl', '_goose') + '.label'
    rel_goose = rel.with_name(label_name)
    
    orig_bin = orig_bin_root / rel.with_suffix('.bin')
    orig_label = orig_label_root / rel_goose    
    out_label = out_label_root / rel_goose    
    
    if out_label.exists():
        return out_label
    out_label.parent.mkdir(parents=True, exist_ok=True)

    # Load decompressed xyz from BIN
    xyz_dec = read_bin_xyz(decomp_bin_path)
    xyz_orig = read_bin_xyz(orig_bin)
    sem_orig, inst_orig = read_label(orig_label)
    if xyz_orig.shape[0] != sem_orig.shape[0]:
        raise ValueError(f"Original point count mismatch: bin {xyz_orig.shape[0]} vs label {sem_orig.shape[0]}")
    xyz_dec = read_bin_xyz(decomp_bin_path)
    xyz_orig = read_bin_xyz(orig_bin)
    sem_orig, inst_orig = read_label(orig_label)
    if xyz_orig.shape[0] != sem_orig.shape[0]:
        raise ValueError(f"Original point count mismatch: bin {xyz_orig.shape[0]} vs label {sem_orig.shape[0]}")

    # Build KD-tree
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_orig)
    tree = o3d.geometry.KDTreeFlann(pcd)

    # Recover labels
    sem_rec = np.empty((xyz_dec.shape[0],), dtype=np.uint32)
    inst_rec = np.empty((xyz_dec.shape[0],), dtype=np.uint32)
    for i, pt in enumerate(xyz_dec):
        [_, idxs, dists] = tree.search_knn_vector_3d(pt, 1)
        dist = np.sqrt(dists[0])
        if dist > threshold:
            raise ValueError(f"No original point within {threshold}m for {decomp_bin_path} index {i} (dist={dist})")
        idx0 = idxs[0]
        sem_rec[i] = sem_orig[idx0]
        inst_rec[i] = inst_orig[idx0]

    # Pack back into uint32 (inst<<16 | sem)
    out_data = (inst_rec.astype(np.uint32) << 16) | sem_rec.astype(np.uint32)
    out_data.tofile(str(out_label))
    return out_label


def main():
    parser = argparse.ArgumentParser(
        description='Restore semantic labels via NN matching (parallel)'
    )
    parser.add_argument('--decomp_bin_root', '-p', type=str, required=True,
                        help='Root of decompressed bins')
    parser.add_argument('--orig_bin_root', '-b', type=str, required=True,
                        help='Root of original BINs')
    parser.add_argument('--orig_label_root', '-l', type=str, required=True,
                        help='Root of original LABELs')
    parser.add_argument('--out_label_root', '-o', type=str, required=True,
                        help='Output root for restored LABELs')
    parser.add_argument('--threshold', '-t', type=float, default=0.01,
                        help='Max distance (m) to match nearest point')
    parser.add_argument('--num_workers', '-n', type=int, default=os.cpu_count(),
                        help='Parallel worker count')
    args = parser.parse_args()

    decomp_bin_root = Path(args.decomp_bin_root)
    orig_bin_root = Path(args.orig_bin_root)
    orig_label_root = Path(args.orig_label_root)
    out_label_root = Path(args.out_label_root)
    threshold = args.threshold
    num_workers = args.num_workers

    decomp_bin_files = list(decomp_bin_root.rglob('*.bin'))
    print(f"Found {len(decomp_bin_files)} decomp bin files under {decomp_bin_root}")
    tasks = [(p, decomp_bin_root, orig_bin_root, orig_label_root, out_label_root, threshold) for p in decomp_bin_files]

    with Pool(processes=num_workers) as pool:
        for out in tqdm(pool.imap(convert_labels_nn, tasks), total=len(tasks), desc="Restoring labels NN"):
            print(f"Restored: {out}")

    print("Label restoration (NN) complete.")

if __name__ == '__main__':
    main()
