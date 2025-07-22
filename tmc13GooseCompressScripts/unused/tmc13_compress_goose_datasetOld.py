import os
import re
import subprocess
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

#  Example Usage:
#  Note: The goose dataset is large so I instead ran this separately for each subdirectory (it will append csv each time instead of overwriting)
# Those commands used are at the bottom of this file

BITSTREAM_PATTERN = re.compile(r"positions bitstream size (?P<bsz>\d+) B \((?P<bpp>[0-9.]+) bpp\)")
ENC_TIME_PATTERN  = re.compile(r"positions processing time.*: (?P<etime>[0-9.]+) s")
DEC_TIME_PATTERN  = re.compile(r"Processing time \(wall\): (?P<dtime>[0-9.]+) s")
# no TOTAL (always 1 frame) and no GPU memory in TMC3 logs

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
    
    # TMC3 compressor settings
    tmc3    = "/home/aniemcz/rellis/compressionTools/TMC13_compressor/mpeg-pcc-tmc13/mpeg-pcc-tmc13/build/tmc3/tmc3"
    cfg_path= "./gpcc.cfg"
    
    parser = argparse.ArgumentParser(
        description="Batch-benchmark TMC13 on the Goose dataset, preserving folder layout."
    )
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of goose .bin files (e.g. ./TMC13/data/goose_examples/lidar)')
    parser.add_argument('--output_root', type=str, default='./analysis',
                        help='Where to write compressed, decompressed data and CSV')
    parser.add_argument('--quant_levels', nargs='+', type=float,
                        default=[], help='Quantization levels')
    parser.add_argument('--input_file_type', type=str, default="ply", choices=['bin', 'ply'], help="Input's file type for TMC13 compressor. Can either be 'bin' or 'ply' although not sure if bin works so should use ply instead")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.output_root)
    
    input_file_ext = args.input_file_type.lstrip('.')
    
    print(f"Input File Type To Search for: {input_file_ext} (You can change this with --input_file_type to either 'bin' or 'ply' although use ply since dont know if tmc13 works with bin)")
    
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
                
        # ─── 1) COMPRESS WITH TMC3 ─────────────────────────────────────────────
        for in_ply in tqdm(all_inputs):
            rel = in_ply.relative_to(data_root)
                            
            # -------Skip if already compressed (cant do this since need it for the results to record) -------
            #already = comp_pres / rel.with_suffix('.bin')
            #if already.exists():
            #    print(f"NOTE: Compressed file of {already} already found, skipping compression")
            #    continue
            # ------------------------------------------
            
            comp_stream = comp_flat / rel.with_suffix('.bin')
            comp_stream.parent.mkdir(parents=True, exist_ok=True)
            run_cmd([
                tmc3,
                '--mode=0',
                f'--config={cfg_path}',
                f'--positionQuantizationScale={q}',
                f'--uncompressedDataPath={in_ply}',
                f'--compressedStreamPath={comp_stream}',
            ])
        
        
        # Extract global metrics        
        m = BITSTREAM_PATTERN.search(out_c)
        comp_bytes  = int(m.group('bsz'))
        bpp         = float(m.group('bpp'))
        m2 = ENC_TIME_PATTERN.search(out_c)
        encode_time = float(m2.group('etime'))
        total_files = 1   # TMC3 always processes 1 frame at a time
        
        # Mirror compressed files into preserved structure, compressed flat will be empty after this
        moved_comp = mirror_and_move(comp_flat, comp_pres, all_inputs_bin, data_root)
        
        # ─── 2) DECOMPRESS WITH TMC3 ────────────────────────────────────────────
        for _, comp_dst in tqdm(moved_comp):
            rel = comp_dst.relative_to(comp_pres)
            
            # -------Skip if already decompressed-------
            #already = decomp_pres / rel.with_suffix('.ply')
            #if already.exists():
            #    print(f"NOTE: Decompressed file of {already} already found, skipping decompression")
            #    continue
            # ------------------------------------------
            
            decomp_ply = decomp_flat / rel.with_suffix('.ply')
            
            decomp_ply.parent.mkdir(parents=True, exist_ok=True)
            run_cmd([
                tmc3,
                '--mode=1',
                f'--compressedStreamPath={comp_dst}',
                f'--reconstructedDataPath={decomp_ply}',
            ])
        
        # Extract decompress time
        m3 = DEC_TIME_PATTERN.search(out_d)
        if not m3:
            raise RuntimeError("Failed to parse TMC3 decode time")
        decode_time   = float(m3.group('dtime'))
        total_files_d = 1
        
        #The output of tmc13 is .ply files so to get filepaths need to update extension from bin to ply
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
                # 'max_gpu_mem_MB': max_mem,  # no GPU mem for TMC3
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

######################################################################################################

pixi run python tmc13_compress_goose_dataset.py   --data_root /scratch/aniemcz/goose-pointcept/quantized_ply_xyz_only_lidar/valEx   --quant_levels 0.0668 0.00521332 0.0001   --output_root /scratch/aniemcz/goose-pointcept/tmc13_decompressed_lidar/valEx

pixi run python tmc13_compress_goose_dataset.py   --data_root /scratch/aniemcz/goose-pointcept/quantized_ply_xyz_only_lidar/val   --quant_levels 0.0668 0.00521332 0.0001   --output_root /scratch/aniemcz/goose-pointcept/tmc13_decompressed_lidar/val

pixi run python tmc13_compress_goose_dataset.py   --data_root /scratch/aniemcz/goose-pointcept/quantized_ply_xyz_only_lidar/trainEx    --quant_levels 0.0668 0.00521332 0.0001   --output_root /scratch/aniemcz/goose-pointcept/tmc13_decompressed_lidar/trainEx

pixi run python tmc13_compress_goose_dataset.py   --data_root /scratch/aniemcz/goose-pointcept/quantized_ply_xyz_only_lidar/train  --quant_levels 0.0668 0.00521332 0.0001   --output_root /scratch/aniemcz/goose-pointcept/tmc13_decompressed_lidar/train


######################################################################################################


The below watch out if you have changed the symlink or are training something that changes the symlink

pixi run python tmc13_compress_goose_dataset.py   --data_root /home/aniemcz/gooseReno/goose-dataset/quantized_ply_xyz_only_lidar/valEx   --quant_levels 0.0668 0.00521332 0.0001   --output_root /home/aniemcz/gooseReno/goose-dataset/tmc13_decompressed_lidar/valEx

pixi run python tmc13_compress_goose_dataset.py   --data_root /home/aniemcz/gooseReno/goose-dataset/quantized_ply_xyz_only_lidar/val   --quant_levels 0.0668 0.00521332 0.0001   --output_root /home/aniemcz/gooseReno/goose-dataset/tmc13_decompressed_lidar/val

pixi run python tmc13_compress_goose_dataset.py   --data_root /home/aniemcz/gooseReno/goose-dataset/quantized_ply_xyz_only_lidar/trainEx    --quant_levels 0.0668 0.00521332 0.0001   --output_root /home/aniemcz/gooseReno/goose-dataset/tmc13_decompressed_lidar/trainEx

pixi run python tmc13_compress_goose_dataset.py   --data_root /home/aniemcz/gooseReno/goose-dataset/quantized_ply_xyz_only_lidar/train  --quant_levels 0.0668 0.00521332 0.0001   --output_root /home/aniemcz/gooseReno/goose-dataset/tmc13_decompressed_lidar/train



'''