import os
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d

'''

python restore_intensity_feature_dataset.py \
  --ply_root ./analysis/Q_8/decompressed \
  --orig_bin_root ./goose-pointcept/lidar \
  --out_bin_root ./analysis/Q_8/decompressed_with_intensity

python restore_intensity_feature_dataset.py \
  --ply_root ./ascii-ply-goose-pointcept-lidar-xyz-only \
  --orig_bin_root ./goose-data-examples/exampleHere \
  --out_bin_root ./decompressed_with_intensity
  
python restore_intensity_feature_dataset.py \
  --ply_root ./goose-data-examples/2023-04-20_campus__0286_1681996776758328417_vls128.ply \
  --orig_bin_root ./goose-data-examples/2023-04-20_campus__0286_1681996776758328417_vls128.bin \
  --out_bin_root ./decompressed_with_intensity

'''

def read_ply_xyz(ply_path: Path):
    pcd = o3d.io.read_point_cloud(str(ply_path), format='ply')  # C++ backend?
    return np.asarray(pcd.points, dtype=np.float32)

def read_bin_intensity(bin_path: Path):
    """
    Reads a .bin point-cloud file (float32 with 4 channels) and returns the intensity column as (N,) array.
    """
    data = np.fromfile(str(bin_path), dtype=np.float32)
    if data.size % 4 != 0:
        raise ValueError(f"Unexpected float count in {bin_path}: got {data.size}")
    points = data.reshape(-1, 4)
    return points[:, 3].astype(np.float32)


def restore_intensity(ply_root: Path, orig_bin_root: Path, out_bin_root: Path):
    """
    Walks all .ply files under ply_root, reads geometry, fetches intensity from matching .bin under orig_bin_root,
    then writes a .bin at the corresponding location under out_bin_root with x,y,z,intensity.
    """
    
    for ply_path in ply_root.rglob('*.ply'):
        rel = ply_path.relative_to(ply_root).with_suffix('.bin')
        orig_bin = orig_bin_root / rel
        out_bin = out_bin_root / rel
        out_bin.parent.mkdir(parents=True, exist_ok=True)

        # Read decompressed geometry and original intensity
        xyz = read_ply_xyz(ply_path)
        intensity = read_bin_intensity(orig_bin)

        if xyz.shape[0] != intensity.shape[0]:
            raise ValueError(
                f"Point count mismatch at {ply_path}\n"
                f"PLY has {xyz.shape[0]} points, original BIN has {intensity.shape[0]}"
            )

        # Stack and write
        print(f"saving to {str(out_bin)}")
        merged = np.hstack((xyz, intensity.reshape(-1, 1)))
        merged.astype(np.float32).tofile(str(out_bin))
        print(f"Restored: {out_bin} ({merged.shape[0]} points)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Restore intensity to decompressed PLY point clouds, outputting BINs with x,y,z,i'
    )
    parser.add_argument(
        '--ply_root', '-p', type=str, required=True,
        help='Root directory of decompressed PLY point clouds'
    )
    parser.add_argument(
        '--orig_bin_root', '-b', type=str, required=True,
        help='Root directory of original BIN dataset (with intensity)'
    )
    parser.add_argument(
        '--out_bin_root', '-o', type=str, required=True,
        help='Output root directory for reconstructed BIN files'
    )
    args = parser.parse_args()

    restore_intensity(
        Path(args.ply_root), Path(args.orig_bin_root), Path(args.out_bin_root)
    )
    print("Intensity restoration complete.")
