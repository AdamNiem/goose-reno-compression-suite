import numpy as np
from pathlib import Path
import argparse
import glob
import random

'''

python sanity_check_bin_ply_geom_only_batch.py \
  --bin_root ./path/to/bin_files \
  --ascii_ply_root ./path/to/ascii_plys \
  --num_samples 10 \
  --seed 42

python sanity_check_bin_ply_geom_only_batch.py \
  --bin_root ./goose-data-examples/exampleHere \
  --ascii_ply_root ./ascii-ply-goose-pointcept-lidar-xyz-only \
  --num_samples 5
  
python sanity_check_bin_ply_geom_only_batch.py \
  --bin_root /scratch/aniemcz/goose-pointcept/lidar \
  --ascii_ply_root /scratch/aniemcz/goose-pointcept/ply_xyz_only_lidar \
  --num_samples 25

'''

def read_bin_xyz(bin_path: Path):
    """Read XYZ coordinates from a binary file"""
    data = np.fromfile(bin_path, dtype=np.float32)
    pts = data.reshape(-1, 4)
    return pts[:, :3]

def read_ascii_ply(ply_path: Path):
    """Read XYZ coordinates from an ASCII PLY file"""
    with open(ply_path, 'r') as f:
        # read header
        line = f.readline().strip()
        if line != 'ply':
            raise ValueError('Not a PLY file')
        
        # parse until end_header
        vertex_count = 0
        while True:
            line = f.readline().strip()
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            if line == 'end_header':
                break
        
        # read points
        pts = np.zeros((vertex_count, 3), dtype=np.float32)
        for i in range(vertex_count):
            parts = f.readline().strip().split()
            if len(parts) >= 3:
                pts[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
        return pts

def compare_arrays(a: np.ndarray, b: np.ndarray):
    """Compare two numpy arrays with tolerance for floating-point precision"""
    if a.shape != b.shape:
        return False, f'Shape mismatch: {a.shape} vs {b.shape}'
    
    # Check for exact match first
    if np.array_equal(a, b):
        return True, 'Exact match'
    
    # Check with tolerance
    if not np.allclose(a, b, atol=1e-6, rtol=1e-6):
        diff = np.abs(a - b)
        maxdiff = diff.max()
        idx = np.unravel_index(np.argmax(diff), a.shape)
        return False, f'Max diff {maxdiff} at index {idx}'
    
    if not np.array_equal(a, b):
        raise Exception("Fail")
        return False, f"Failed array_equal check. Are not exactly the same shape and same values"
    
    return True, 'Match within tolerance'

def find_all_files(root_dir: Path, pattern: str):
    """Recursively find all files matching pattern under root_dir"""
    return [Path(p) for p in glob.glob(str(root_dir / '**' / pattern), recursive=True)]

def main():
    parser = argparse.ArgumentParser(description='Sanity check BIN vs ASCII PLY')
    
    # Single file mode
    parser.add_argument('--bin', type=str, help='Single original .bin file path')
    parser.add_argument('--ply_ascii', type=str, help='Single ASCII PLY file path')
    
    # Batch mode
    parser.add_argument('--bin_root', type=str, help='Root directory for .bin files')
    parser.add_argument('--ascii_ply_root', type=str, help='Root directory for ASCII PLY files')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of random samples to test (batch mode)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    
    args = parser.parse_args()
    random.seed(args.seed)

    # Single file mode
    if args.bin and args.ply_ascii:
        p_bin = Path(args.bin)
        p_ascii = Path(args.ply_ascii)

        print(f"\nTesting single file: {p_bin.name}")
        try:
            xyz_bin = read_bin_xyz(p_bin)
            xyz_ascii = read_ascii_ply(p_ascii)
            ok, msg = compare_arrays(xyz_bin, xyz_ascii)
            print(f'BIN vs ASCII PLY: {msg}')
            if ok:
                print('Formats match')
            else:
                print('Discrepancies found.')
        except Exception as e:
            print(f"ERROR: {str(e)}")
        return

    # Batch mode
    if not (args.bin_root and args.ascii_ply_root):
        parser.error("For batch mode, both bin_root and ascii_ply_root must be provided")
    
    bin_root = Path(args.bin_root)
    ascii_ply_root = Path(args.ascii_ply_root)
    
    # Find all bin files
    bin_files = find_all_files(bin_root, "*.bin")
    print(f"Found {len(bin_files)} .bin files")
    
    if not bin_files:
        print("No .bin files found. Exiting.")
        return
    
    # Randomly sample files
    if args.num_samples > 0 and args.num_samples < len(bin_files):
        sample_files = random.sample(bin_files, args.num_samples)
    else:
        sample_files = bin_files
        print(f"Testing all {len(bin_files)} files")
    
    print(f"\nTesting {len(sample_files)} samples:")
    
    results = []
    for bin_path in sample_files:
        # Get relative path
        rel_path = bin_path.relative_to(bin_root)
        
        # Construct expected ply path
        ply_path = ascii_ply_root / rel_path.with_suffix('.ply')
        
        # Check if PLY file exists
        if not ply_path.exists():
            print(f"  WARNING: ASCII PLY not found: {ply_path}")
            results.append({
                'file': str(rel_path),
                'status': 'MISSING',
                'message': f'PLY file not found: {ply_path}'
            })
            continue
        
        # Read and compare
        try:
            xyz_bin = read_bin_xyz(bin_path)
            xyz_ascii = read_ascii_ply(ply_path)
            ok, msg = compare_arrays(xyz_bin, xyz_ascii)
            
            status = "PASS" if ok else "FAIL"
            results.append({
                'file': str(rel_path),
                'status': status,
                'message': msg
            })
            print(f"  {status} - {rel_path}")
            
        except Exception as e:
            results.append({
                'file': str(rel_path),
                'status': 'ERROR',
                'message': str(e)
            })
            print(f"  ERROR processing {rel_path}: {str(e)}")
    
    # Print summary
    print("\nTest Summary:")
    print(f"Files tested: {len(results)}")
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] == 'FAIL')
    errors = sum(1 for r in results if r['status'] == 'ERROR')
    missing = sum(1 for r in results if r['status'] == 'MISSING')
    
    print(f"PASS: {passed}, FAIL: {failed}, ERRORS: {errors}, MISSING: {missing}")
    
    # Print failure details
    if failed > 0 or errors > 0 or missing > 0:
        print("\nDetails for non-passing files:")
        for r in results:
            if r['status'] != 'PASS':
                print(f"\nFile: {r['file']}")
                print(f"  Status: {r['status']}")
                print(f"  Message: {r['message']}")

if __name__ == '__main__':
    main()