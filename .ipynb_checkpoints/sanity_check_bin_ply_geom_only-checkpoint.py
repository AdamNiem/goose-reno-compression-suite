import numpy as np
from pathlib import Path
import argparse

'''

python sanity_check_bin_ply_geom_only.py \
  --bin path/to/original.bin \
  --ply_ascii path/to/stripped_ascii.ply \
  --ply_bin path/to/stripped_binary.ply
  
python sanity_check_bin_ply_geom_only.py \
  --bin ./goose-data-examples/exampleHere/trainEx/2023-04-20_campus__0286_1681996776758328417_vls128.bin \
  --ply_ascii ./ascii-ply-goose-pointcept-lidar-xyz-only/trainEx/2023-04-20_campus__0286_1681996776758328417_vls128.ply \
  --ply_bin ./lil-endian-ply-goose-pointcept-lidar-xyz-only/trainEx/2023-04-20_campus__0286_1681996776758328417_vls128.ply


'''

def read_bin_xyz(bin_path: Path):
    data = np.fromfile(bin_path, dtype=np.float32)
    pts = data.reshape(-1, 4)
    return pts[:, :3]


def read_ascii_ply(ply_path: Path):
    with open(ply_path, 'r') as f:
        # read header
        line = f.readline().strip()
        if line != 'ply':
            raise ValueError('Not a PLY file')
        # parse until end_header
        while True:
            line = f.readline().strip()
            if line.startswith('format'):
                if 'ascii' not in line:
                    raise ValueError('Not ASCII PLY')
            if line == 'end_header':
                break
        # read points
        pts = []
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
        return np.array(pts, dtype=np.float32)


def read_binary_ply(ply_path: Path):
    with open(ply_path, 'rb') as f:
        # read header
        header = b''
        while True:
            line = f.readline()
            header += line
            if line.strip() == b'end_header':
                break
        # after header, binary floats
        data = f.read()
        coords = np.frombuffer(data, dtype=np.float32)
        pts = coords.reshape(-1, 3)
        return pts


def compare_arrays(a: np.ndarray, b: np.ndarray):
    if a.shape != b.shape:
        print("FAILED SHAPE COMPARE")
        return False, f'Shape mismatch: {a.shape} vs {b.shape}'
    if not np.allclose(a, b, atol=0, rtol=0):
        diff = np.abs(a - b)
        maxdiff = diff.max()
        idx = np.unravel_index(np.argmax(diff), a.shape)
        print("FAILED ALLCLOSE")
        return False, f'Max diff {maxdiff} at index {idx}'
    
    if not np.array_equal(a, b):
        print("FAILED ARRAY EQUAL")
        return False, f"Failed array_equal check. Are not exactly the same shape and same values"
    
    return True, 'Exact match'


def main():
    parser = argparse.ArgumentParser(description='Sanity check BIN vs ASCII PLY vs BINARY PLY')
    parser.add_argument('--bin', type=str, required=True, help='Original .bin file path')
    parser.add_argument('--ply_ascii', type=str, required=True, help='ASCII PLY file path')
    parser.add_argument('--ply_bin', type=str, required=True, help='Binary PLY file path')
    args = parser.parse_args()

    p_bin = Path(args.bin)
    p_ascii = Path(args.ply_ascii)
    p_binply = Path(args.ply_bin)

    xyz_bin = read_bin_xyz(p_bin)
    xyz_ascii = read_ascii_ply(p_ascii)
    xyz_binply = read_binary_ply(p_binply)

    ok1, msg1 = compare_arrays(xyz_bin, xyz_ascii)
    ok2, msg2 = compare_arrays(xyz_bin, xyz_binply)

    print(f'BIN vs ASCII PLY: {msg1}')
    print(f'BIN vs BINARY PLY: {msg2}')
    if ok1 and ok2:
        print('All formats match exactly')
    else:
        print('Discrepancies found.')

if __name__ == '__main__':
    main()
