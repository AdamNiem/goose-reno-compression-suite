import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import multiprocessing

'''
# Create quantized PLY dataset directly from BIN files using fixed quantization
# Parallel processing with multiprocessing.Pool
# Usage:
# python create_quantized_ascii_ply_xyz_only_dataset_parallel.py \
#   --input_root ./goose-pointcept/lidar \
#   --output_root ./quantized_ply_lidar \
#   --num_workers 8

python create_quantized_ascii_ply_xyz_only_dataset_parallel.py \
  --input_root /scratch/aniemcz/goose-pointcept/lidar \
  --output_root /scratch/aniemcz/goose-pointcept/quantized_ply_xyz_only_lidar
'''

def quantize(coords: np.ndarray) -> np.ndarray:
    '''quantize point cloud coords to 18 bit (1mm) precision'''
    # scale to 1mm, offset to avoid negatives, deduplicate
    coords = np.round(coords / 0.001) + 131072
    coords = np.unique(coords, axis=0)
    return coords


def convert_file(args):
    """Worker function for parallel processing"""
    bin_path, input_root, output_root = args
    # Compute relative path and PLY output path
    rel_path = bin_path.relative_to(input_root).with_suffix('.ply')
    out_path = output_root / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read binary data as float32, reshape into N x 4 and extract xyz
    data = np.fromfile(bin_path, dtype=np.float32)
    if data.size % 4 != 0:
        raise ValueError(f"Unexpected float count in {bin_path}: {data.size}")
    pts = data.reshape(-1, 4)
    coords = pts[:, :3]

    # Quantize + deduplicate
    coords_q = quantize(coords).astype(np.float32)
    num_pts = coords_q.shape[0]

    # Write ASCII PLY
    with open(out_path, 'w') as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {num_pts}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")
        for x, y, z in coords_q:
            ply_file.write(f"{x} {y} {z}\n")

    return out_path


def process_files(bin_files, input_root, output_root, num_workers):
    """Process files in parallel"""
    # Create argument tuples for workers
    tasks = [(f, input_root, output_root) for f in bin_files]
    
    # Use multiprocessing Pool
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(convert_file, tasks),
            total=len(tasks),
            desc="Quantizing and converting files"
        ))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create quantized PLY dataset from BIN files with fixed quantization"
    )
    parser.add_argument(
        '--input_root', '-i', type=str, required=True,
        help='Root directory of original .bin files'
    )
    parser.add_argument(
        '--output_root', '-o', type=str, required=True,
        help='Directory where quantized PLYs will be created'
    )
    parser.add_argument(
        '--num_workers', '-n', type=int, default=os.cpu_count(),
        help='Number of parallel workers (default: CPU count)'
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    # Gather all BIN files
    bin_files = list(input_root.rglob('*.bin'))
    print(f"Found {len(bin_files)} BIN files under {input_root}")

    results = process_files(bin_files, input_root, output_root, args.num_workers)
    
    print(f"Completed writing {len(results)} Quantized XYZ-only ASCII PLY files at:", output_root)