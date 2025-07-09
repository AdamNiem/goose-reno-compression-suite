import os
import argparse
from pathlib import Path
import numpy as np

'''
python create_xyz_only_dataset.py \
  --input_root ./goose-pointcept/lidar \
  --output_root ./goose-pointcept-lidar-xyz-only

python create_lil_endian_ply_dataset.py \
  --input_root ./goose-data-examples/exampleHere \
  --output_root ./lil-endian-ply-goose-pointcept-lidar-xyz-only
'''

def strip_xyz_to_ply(input_root: Path, output_root: Path):
    """
    Recursively reads all .bin files under input_root, extracts x,y,z,
    and writes them as BINARY little-endian PLY files under output_root preserving structure.
    This ensures no precision loss compared to ASCII formats.
    """
    for bin_path in input_root.rglob('*.bin'):
        # Compute relative path and change suffix to .ply
        rel_path = bin_path.relative_to(input_root).with_suffix('.ply')
        out_path = output_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Read binary data as float32, reshape into N x 4
        data = np.fromfile(bin_path, dtype=np.float32)
        if data.size % 4 != 0:
            raise ValueError(f"Unexpected float count in {bin_path}: {data.size}")
        points = data.reshape(-1, 4)
        xyz = points[:, :3].astype(np.float32)
        num_pts = xyz.shape[0]

        # Write BINARY little-endian PLY
        with open(out_path, 'wb') as ply_file:
            # Header (ASCII)
            header = []
            header.append("ply")
            header.append("format binary_little_endian 1.0")
            header.append(f"element vertex {num_pts}")
            header.append("property float x")
            header.append("property float y")
            header.append("property float z")
            header.append("end_header")
            ply_file.write("\n".join(header).encode('ascii'))
            ply_file.write(b"\n")
            # Body: write raw float32 byteso
            ply_file.write(xyz.tobytes())

        print(f"Wrote {out_path} ({num_pts} vertices, binary PLY)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Strip intensity and write x,y,z-only BINARY PLY point clouds"
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
    print("Completed writing XYZ-only BINARY PLY dataset at:", output_root)
