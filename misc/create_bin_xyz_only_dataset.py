import os
import argparse
from pathlib import Path
import numpy as np

'''
python create_xyz_only_dataset.py \
  --input_root ./goose-pointcept/lidar \
  --output_root ./goose-pointcept-lidar-xyz-only
  
python create_xyz_only_dataset.py \
  --input_root ./goose-data-examples/exampleHere \
  --output_root ./goose-pointcept-lidar-xyz-only
'''

def strip_xyz(input_root: Path, output_root: Path):
    """
    Recursively copies all .bin files from input_root to output_root,
    stripping each point cloud to only its x,y,z components.
    Preserves directory structure.
    """
    # Walk through all .bin files under input_root
    for bin_path in input_root.rglob('*.bin'):
        # Compute relative path and prepare output file path
        rel_path = bin_path.relative_to(input_root)
        out_path = output_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Read floats from binary and reshape to (-1, num_features)
        data = np.fromfile(bin_path, dtype=np.float32)
        if data.size % 4 != 0:
            raise ValueError(f"Unexpected number of floats in {bin_path}: {data.size}")
        points = data.reshape(-1, 4)

        # Extract only x,y,z
        xyz = points[:, :3]

        # Write new binary containing only xyz floats
        xyz.astype(np.float32).tofile(out_path)
        print(f"Wrote {out_path} ({xyz.shape[0]} points)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Strip intensity/extra channels and retain only x,y,z in .bin point clouds"
    )
    parser.add_argument(
        '--input_root', '-i',
        type=str,
        required=True,
        help='Root directory of original Goose lidar .bin files (e.g. ./goose/lidar)'
    )
    parser.add_argument(
        '--output_root', '-o',
        type=str,
        required=True,
        help='Directory where stripped .bin dataset will be created'
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    strip_xyz(input_root, output_root)
    print("Completed stripping intensity; new dataset at:", output_root)
