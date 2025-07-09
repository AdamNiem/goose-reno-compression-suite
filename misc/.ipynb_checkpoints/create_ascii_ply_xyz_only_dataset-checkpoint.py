import os
import argparse
from pathlib import Path
import numpy as np

from tqdm import tqdm

'''
python create_ascii_ply_xyz_only_dataset.py \
  --input_root ./goose-pointcept/lidar \
  --output_root ./goose-pointcept-lidar-xyz-only

python create_ascii_ply_xyz_only_dataset.py \
  --input_root ./goose-data-examples/exampleHere \
  --output_root ./ascii-ply-goose-pointcept-lidar-xyz-only
  
python create_ascii_ply_xyz_only_dataset.py \
  --input_root /scratch/aniemcz/goose-pointcept/lidar \
  --output_root /scratch/aniemcz/goose-pointcept/ply_xyz_only_lidar
'''

def strip_xyz_to_ply(input_root: Path, output_root: Path):
    """
    Recursively reads all .bin files under input_root, extracts x,y,z,
    and writes them as ASCII PLY files under output_root preserving structure.
    """
    for bin_path in tqdm(input_root.rglob('*.bin')):
        # Compute relative path and PLY output path
        rel_path = bin_path.relative_to(input_root).with_suffix('.ply')
        out_path = output_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Read binary data as float32, reshape into N x 4
        data = np.fromfile(bin_path, dtype=np.float32)
        if data.size % 4 != 0:
            raise ValueError(f"Unexpected float count in {bin_path}: {data.size}")
        points = data.reshape(-1, 4)
        xyz = points[:, :3]
        num_pts = xyz.shape[0]

        # Write ASCII PLY
        with open(out_path, 'w') as ply_file:
            # Header
            ply_file.write("ply\n")
            ply_file.write("format ascii 1.0\n")
            ply_file.write(f"element vertex {num_pts}\n")
            ply_file.write("property float x\n")
            ply_file.write("property float y\n")
            ply_file.write("property float z\n")
            ply_file.write("end_header\n")
            # Body
            for x, y, z in xyz:
                ply_file.write(f"{x} {y} {z}\n")
        #print(f"Wrote {out_path} with {num_pts} vertices")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Strip intensity and write x,y,z-only ASCII PLY point clouds"
    )
    parser.add_argument(
        '--input_root', '-i', type=str, required=True,
        help='Root directory of original Goose lidar .bin files (e.g. ./goose/lidar)'
    )
    parser.add_argument(
        '--output_root', '-o', type=str, required=True,
        help='Directory where XYZ-only PLY dataset will be created'
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    strip_xyz_to_ply(input_root, output_root)
    print("Completed writing XYZ-only ASCII PLY dataset at:", output_root)
