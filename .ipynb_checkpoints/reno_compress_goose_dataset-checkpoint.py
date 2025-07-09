import os
import re
import subprocess
import argparse
import pandas as pd
from pathlib import Path

#  Example Usage:
#  Note: The goose dataset is large so I instead ran this separately for each subdirectory (it will append csv each time instead of overwriting)
#
#  python reno_compress_goose_dataset.py \
#    --data_root ./RENO/data/goose_examples/lidar \
#    --ckpt ./RENO/model/Goose/ckpt.pt \
#    --output_root ./analysis \
#    --quant_levels 8 64 512 \
#    --input_file_type ply
#
#  For one file only to test it out:
#
#  python reno_compress_goose_dataset.py \
#  --data_root ./goose-data-examples/exampleHere/2023-04-20_campus__0286_1681996776758328417_vls128.bin \
#  --ckpt ./RENO/model/Goose/ckpt.pt \
#  --output_root ./analysis

'''
  python reno_compress_goose_dataset.py \
  --data_root ./goose-data-examples/exampleHere \
  --ckpt ./RENO/model/Goose/ckpt.pt \
  --output_root ./analysis
  
  python reno_compress_goose_dataset.py \
  --data_root ./goose-data-examples/exampleHere \
  --ckpt ./RENO/model/Goose/ckpt.pt \
   --quant_levels 8 64 512 \
  --output_root ./analysis
  
  python reno_compress_goose_dataset.py \
  --data_root /scratch/aniemcz/goose-pointcept/ply_xyz_only_lidar \
  --ckpt ./RENO/model/Goose/ckpt.pt \
  --quant_levels 8 64 512 \
  --output_root /scratch/aniemcz/goose-pointcept/reno_decompressed_lidar
'''

'''
  python reno_compress_goose_dataset.py \
  --data_root /scratch/aniemcz/goose-pointcept/ply_xyz_only_lidar/train \
  --ckpt ./RENO/model/Goose/ckpt.pt \
  --quant_levels 8 64 512 \
  --output_root /scratch/aniemcz/goose-pointcept/reno_decompressed_lidar/train
  
  python reno_compress_goose_dataset.py \
  --data_root /scratch/aniemcz/goose-pointcept/ply_xyz_only_lidar/trainEx \
  --ckpt ./RENO/model/Goose/ckpt.pt \
  --quant_levels 8 64 512 \
  --output_root /scratch/aniemcz/goose-pointcept/reno_decompressed_lidar/trainEx
'''

# Regex for extracting global metrics from RENO output
BPP_PATTERN = re.compile(r"Avg\. Bpp:(?P<bpp>[0-9.]+)")
ENC_TIME_PATTERN = re.compile(r"Encode time:(?P<etime>[0-9.]+)")
MEM_PATTERN = re.compile(r"Max GPU Memory:(?P<mem>[0-9.]+)MB")
DEC_TIME_PATTERN = re.compile(r"Decode Time:(?P<dtime>[0-9.]+)")
TOTAL_PATTERN = re.compile(r"Total:\s*(?P<total>\d+)")

'''
def run_cmd(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)    
    print(proc.stdout, end='')
    if proc.stderr:
        print(proc.stderr, end='', file=sys.stderr)
        
    if proc.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed:\n{proc.stderr}")
    return proc.stdout  # Needed for regex extraction
'''

def run_cmd(cmd):
    # Launch the process, merging stderr into stdout
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    full_output = []
    # Read lines as they arrive
    for line in proc.stdout:
        print(line, end='')        # real-time echo
        full_output.append(line)   # buffer for return

    # Wait for exit code
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed with exit code {proc.returncode}")

    return ''.join(full_output)


def mirror_and_move(src_flat: Path, dst_root: Path, files, src_root: Path):
    """
    Move files from a flat directory into a mirrored folder structure under dst_root,
    matching the relative paths in `files` (list of input Paths relative to src_root).
    """
    moved = []
    for f in files:
        rel = f.relative_to(src_root)
        src_file = src_flat / f.name
        dst_file = dst_root / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        src_file.replace(dst_file)
        moved.append((f, dst_file))
    return moved


def main():
    parser = argparse.ArgumentParser(
        description="Batch-benchmark RENO on the Goose dataset, preserving folder layout."
    )
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of goose .bin files (e.g. ./RENO/data/goose_examples/lidar)')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the RENO checkpoint')
    parser.add_argument('--output_root', type=str, default='./analysis',
                        help='Where to write compressed, decompressed data and CSV')
    parser.add_argument('--quant_levels', nargs='+', type=int,
                        default=[8,16,32,64,128,256,512], help='Quantization levels')
    parser.add_argument('--input_file_type', type=str, default="ply", choices=['bin', 'ply'], help="Input's file type for RENO compressor. Can either be 'bin' or 'ply'")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.output_root)
    ckpt = Path(args.ckpt)
    
    input_file_ext = args.input_file_type.lstrip('.')
    
    print(f"Input File Type To Search for: {input_file_ext} (You can change this with --input_file_type to either 'bin' or 'ply')")

    # Gather all input files
    all_inputs = list(data_root.rglob(f'*.{input_file_ext}'))
    
     # Gather all input files (either .bin or .ply)
    all_inputs = list(data_root.rglob(f'*.{input_file_ext}'))
    print(f"{len(all_inputs)} files found of type {input_file_ext}")
    
    if len(all_inputs) == 0:
        raise Exception(f"Could not find any files of type {input_file_ext} at {data_root}")

    # But compression always emits .bin, so build a companion list of .bin paths for mirroring
    if input_file_ext == 'ply':
        all_inputs_bin = [p.with_suffix('.bin') for p in all_inputs]
    else:
        all_inputs_bin = all_inputs
    
    # Prepare CSV records
    records = []

    for q in args.quant_levels:
        print(f"\n=== Quantization: {q} ===")
        temp_root = out_root / f"Q_{q}"
        comp_flat = temp_root / 'flat_compressed'
        decomp_flat = temp_root / 'flat_decompressed'
        comp_pres = temp_root / 'compressed'
        decomp_pres = temp_root / 'decompressed'
        comp_flat.mkdir(parents=True, exist_ok=True)
        decomp_flat.mkdir(parents=True, exist_ok=True)
        
        #NOTE: Added new py files to RENO just to prevent file extension stacking (ex: .bin -> .bin.ply or .bin -> .bin.bin)

        # Run compression on all files at once
        glob_pattern = str(data_root / '**' / f'*.{input_file_ext}')
        out_c = run_cmd([
            'python', str(Path(__file__).parent / 'RENO/compressNew.py'),
            '--input_glob', glob_pattern,
            '--output_folder', str(comp_flat),
            '--ckpt', str(ckpt),
            '--posQ', str(q)
        ])
        # Extract global metrics
        bpp = float(BPP_PATTERN.search(out_c).group('bpp'))
        encode_time = float(ENC_TIME_PATTERN.search(out_c).group('etime'))
        max_mem = float(MEM_PATTERN.search(out_c).group('mem'))
        total_files   = int(TOTAL_PATTERN.search(out_c).group('total'))
        
        # Mirror compressed files into preserved structure, compressed flat will be empty after this
        moved_comp = mirror_and_move(comp_flat, comp_pres, all_inputs_bin, data_root)

        # Run decompression on all compressed bins
        glob_comp = str(comp_pres / '**' / '*.bin')
        
        print(f"decomp glob is {glob_comp}")
        
        out_d = run_cmd([
            'python', str(Path(__file__).parent / 'RENO/decompressToBin.py'),
            '--input_glob', glob_comp,
            '--output_folder', str(decomp_flat),
            '--ckpt', str(ckpt)
        ])
        decode_time = float(DEC_TIME_PATTERN.search(out_d).group('dtime'))
        total_files_d = int(TOTAL_PATTERN.search(out_d).group('total'))
        
        #The output of reno is .ply files so to get filepaths need to update extension from bin to ply
        #moved_comp is list of tuples with each tuple containing posix / path object of file in original data location
        #So we get posix / path object out of tuple, swap out extension for .ply, 
        #and then make the file paths as list of strings instead of list of path objects
        decomp_paths = [m[0].with_suffix('.ply') for m in moved_comp]
    
        # Mirror decompressed files back
        moved_decomp = mirror_and_move(decomp_flat, decomp_pres,
                                       decomp_paths, data_root)

        # Per-file size & ratio
        orig_inputs = all_inputs
        for orig, (_, comp_dst), (_, decomp_dst) in zip(orig_inputs, moved_comp, moved_decomp):
            orig_size = orig.stat().st_size
            comp_size = comp_dst.stat().st_size
            decomp_size = decomp_dst.stat().st_size
            rec = {
                'rel_path': str(orig.relative_to(data_root)),
                'full_path': str(orig.resolve()),
                'quant': q,
                'batch_total_files': total_files, # from compress (should be same as decompress just a sanity check)
                'batch_total_files_dec': total_files_d, # from decompress
                'avg_bpp_all': bpp,
                'encode_time_all_s': encode_time,
                'decode_time_all_s': decode_time,
                'max_gpu_mem_MB': max_mem,
                'orig_bytes': orig_size,
                'comp_bytes': comp_size,
                'decomp_bytes': decomp_size,
                'ratio': orig_size / comp_size
            }
            records.append(rec)
         
        #remove the comp flat and decomp flat folders since they are no longer needed after the move
        comp_flat.rmdir()
        decomp_flat.rmdir()

    # Write parquet
    '''
    df = pd.DataFrame(records)
    parquet_path = out_root / 'compression_benchmark.parquet'
    df.to_parquet(parquet_path, index=False)
    print(f"Results saved to {parquet_path}")
    '''
    
    # Write csv
    df = pd.DataFrame(records)
    csv_path = out_root / 'compression_benchmark.csv'
    
    # Check if file exists and append if it does
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        df = pd.concat([existing_df, df], ignore_index=True)
        print(f"Appending to existing CSV at {csv_path}")
    
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


if __name__ == '__main__':
    main()

    
    
'''
All commands to run to preprocess entire goose dataset

test  testEx  train  trainEx  val  valEx

pixi run python reno_compress_goose_dataset.py   --data_root /home/aniemcz/gooseReno/goose-dataset/ply_xyz_only_lidar/valEx   --ckpt ./RENO/model/Goose/ckpt.pt   --quant_levels 8 64 512   --output_root /home/aniemcz/gooseReno/goose-dataset/reno_decompressed_lidar/valEx

pixi run python reno_compress_goose_dataset.py   --data_root /home/aniemcz/gooseReno/goose-dataset/ply_xyz_only_lidar/val   --ckpt ./RENO/model/Goose/ckpt.pt   --quant_levels 8 64 512   --output_root /home/aniemcz/gooseReno/goose-dataset/reno_decompressed_lidar/val

pixi run python reno_compress_goose_dataset.py   --data_root /home/aniemcz/gooseReno/goose-dataset/ply_xyz_only_lidar/trainEx   --ckpt ./RENO/model/Goose/ckpt.pt   --quant_levels 8 64 512   --output_root /home/aniemcz/gooseReno/goose-dataset/reno_decompressed_lidar/trainEx

pixi run python reno_compress_goose_dataset.py   --data_root /home/aniemcz/gooseReno/goose-dataset/ply_xyz_only_lidar/train   --ckpt ./RENO/model/Goose/ckpt.pt   --quant_levels 8 64 512   --output_root /home/aniemcz/gooseReno/goose-dataset/reno_decompressed_lidar/train

'''