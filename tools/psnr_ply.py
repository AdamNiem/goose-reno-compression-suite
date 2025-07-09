import open3d as o3d
import numpy as np
import sys

def load_points_from_ply(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    return np.asarray(pcd.points)

def compute_mse(a, b):
    return np.mean((a - b) ** 2)

def compute_psnr(mse, max_val=1.0):
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_val ** 2) / mse)

def main(original_ply, decompressed_ply):
    pts_orig = load_points_from_ply(original_ply)
    pts_decomp = load_points_from_ply(decompressed_ply)

    if pts_orig.shape != pts_decomp.shape:
        print("Error: Point clouds do not match in shape.")
        sys.exit(1)

    mse = compute_mse(pts_orig, pts_decomp)

    # Estimate dynamic range from original data
    max_range = np.max(pts_orig) - np.min(pts_orig)
    psnr = compute_psnr(mse, max_val=max_range)

    print(f"MSE: {mse:.6f}")
    print(f"PSNR: {psnr:.2f} dB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("original_ply", help="Path to original PLY file")
    parser.add_argument("decompressed_ply", help="Path to decompressed PLY file")
    args = parser.parse_args()

    main(args.original_ply, args.decompressed_ply)

