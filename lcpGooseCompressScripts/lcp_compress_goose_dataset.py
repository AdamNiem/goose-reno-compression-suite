import os
import re
import subprocess
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# LCP output patterns
RATIO_PATTERN      = re.compile(r"compression ratio = (?P<ratio>[0-9.]+)")
CTIME_PATTERN      = re.compile(r"compression time = (?P<ctime>[0-9.]+)")
D_TIME_PATTERN     = re.compile(r"decompression time = (?P<dtime>[0-9.]+)")
AXIS_STATS_PATTERN = re.compile(
    r"statistics of (?P<axis>[xyz])\s+Min=(?P<min>[0-9\.\-E]+), "
    r"Max=(?P<max>[0-9\.\-E]+), range=(?P<range>[0-9\.\-E]+)"
)
ABS_ERR_PATTERN    = re.compile(r"Max absolute error = (?P<abs_err>[0-9\.\-E]+)")
REL_ERR_PATTERN    = re.compile(r"Max relative error = (?P<rel_err>[0-9\.\-E]+)")
PSNR_PATTERN       = re.compile(r"PSNR = (?P<psnr>[0-9\.\-E]+)")
NRMSE_PATTERN      = re.compile(r"NRMSE=\s*(?P<nrmse>[0-9\.\-E]+)")

'''
python lcp_compress_goose_dataset.py \
  --data_root /scratch/aniemcz/goose-pointcept/dat_xyz_only_lidar \
  --ply_root /scratch/aniemcz/goose-pointcept/ply_xyz_only_lidar \
  --output_root /scratch/aniemcz/goose-pointcept/lcp_compression_results
'''

def run_cmd(cmd):
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    output_lines = []
    for line in proc.stdout:
        print(line, end='')
        output_lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command {cmd} failed with exit code {proc.returncode}")
    return ''.join(output_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Batch-benchmark LCP on the Goose dataset"
    )
    parser.add_argument('--data_root',   type=str, required=True,
                        help='Root of x/y/z .dat files (from bin_to_dat)')
    parser.add_argument('--ply_root',    type=str, required=True,
                        help='Root of original .ply files for size comparison')
    parser.add_argument('--output_root', type=str, default='./analysis',
                        help='Where to write results')
    parser.add_argument('--quant_levels', nargs='+', type=float,
                        default=[0.689, 0.2364, 0.085901831, 1e-1, 1e-2, ], help='Quantization scales (eb)')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ply_root  = Path(args.ply_root)
    out_root  = Path(args.output_root)
    lcp       = "/home/aniemcz/rellis/compressionTools/lcp_compressor/LCP/compiledExecutable/bin/lcp"

    # find all x‐dat files
    x_files = sorted(data_root.rglob('*_x.dat'))
    print(f"Found {len(x_files)} scans")

    records = []
    for q in args.quant_levels:
        print(f"\n=== Quantization EB: {q} ===")
        tmp        = out_root / f"EB_{q}"
        comp_pres  = tmp / 'compressed'
        decomp_pres= tmp / 'decompressed'
        comp_pres.mkdir(parents=True, exist_ok=True)
        decomp_pres.mkdir(parents=True, exist_ok=True)

        # process each scan individually
        for x in tqdm(x_files, desc=f"EB={q}"):
            # build file paths
            rel   = x.relative_to(data_root)
            rel_str = str(rel)
            y = Path(str(x).replace('_x.dat', '_y.dat'))
            z = Path(str(x).replace('_x.dat', '_z.dat'))
            
            # -----Skip if already decompressed--------
            already_x = decomp_pres / rel_str
            if already_x.exists():
                print(f"NOTE: Decompressed file of {already_x} already found, skipping")
                continue
            # -----------------------------------------
            
            out_lcp = tmp / 'flat_compressed' / rel.with_suffix('.lcp')
            out_lcp.parent.mkdir(parents=True, exist_ok=True)

            # read number of points
            x_arr = np.fromfile(str(x), dtype=np.float32)
            N     = x_arr.size
            y_arr = np.fromfile(str(y), dtype=np.float32)
            z_arr = np.fromfile(str(z), dtype=np.float32)
            #small sanity check
            if (x_arr.shape[0] != y_arr.shape[0]) or (x_arr.shape[0] != z_arr.shape[0]):
                raise Exception(f"ERROR: The file with rel {rel_str} dont have same num pts for xyz")
                
            # prepare decompressed‐flat paths
            out_x = tmp / 'flat_decompressed' / Path(rel_str)
            out_y = tmp / 'flat_decompressed' / Path(rel_str.replace('_x.dat', '_y.dat'))
            out_z = tmp / 'flat_decompressed' / Path(rel_str.replace('_x.dat', '_z.dat'))
            out_x.parent.mkdir(parents=True, exist_ok=True)

            # 1) compress + decompress in one call
            cmd_c = [
                lcp,
                '-i', str(x), str(y), str(z),
                '-z', str(out_lcp),
                '-o', str(out_x), str(out_y), str(out_z),
                '-1', str(N),
                '-eb', str(q), '-bt', '1', '-a'
            ]
            out_c = run_cmd(cmd_c)

            # parse compress metrics
            ratio       = float(RATIO_PATTERN.search(out_c).group('ratio'))
            encode_time = float(CTIME_PATTERN.search(out_c).group('ctime'))
            decode_time = float(D_TIME_PATTERN.search(out_c).group('dtime'))
            axis_stats  = {a: {} for a in 'xyz'}
            for m in AXIS_STATS_PATTERN.finditer(out_c):
                axis_stats[m.group('axis')] = {
                    'min':   float(m.group('min')),
                    'max':   float(m.group('max')),
                    'range': float(m.group('range'))
                }
            abs_errs = ABS_ERR_PATTERN.findall(out_c)
            rel_errs = REL_ERR_PATTERN.findall(out_c)
            psnrs    = PSNR_PATTERN.findall(out_c)
            nrmse    = NRMSE_PATTERN.search(out_c).group('nrmse')

            # mirror compressed → compressed/
            dst_c = comp_pres / rel.with_suffix('.lcp')
            dst_c.parent.mkdir(parents=True, exist_ok=True)
            out_lcp.replace(dst_c)

            # mirror decompressed → decompressed/
            for src, suffix in [(out_x, '_x.dat'),
                                (out_y, '_y.dat'),
                                (out_z, '_z.dat')]:
                dst_rel = rel_str.replace('_x.dat', suffix)
                dst = decomp_pres / dst_rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.replace(dst)

            # lookup original ply size
            ply_file      = ply_root /  Path(rel_str.replace('_x.dat', '.ply'))
            orig_bytes_ply= ply_file.stat().st_size

            # record row
            records.append({
                'rel_path':          str(rel),
                'eb':                q,
                'compression_ratio': ratio,
                'encode_time_s':     encode_time,
                'decode_time_s':     decode_time,
                'orig_bytes_dat':    x.stat().st_size,
                'orig_bytes_ply':    orig_bytes_ply,
                'comp_bytes':        dst_c.stat().st_size,
                'min_x':             axis_stats['x']['min'],
                'max_x':             axis_stats['x']['max'],
                'range_x':           axis_stats['x']['range'],
                'abs_err_x':         float(abs_errs[0]),
                'rel_err_x':         float(rel_errs[0]),
                'psnr_x':            float(psnrs[0]),
                'nrmse_x':           float(nrmse),
                'min_y':             axis_stats['y']['min'],
                'max_y':             axis_stats['y']['max'],
                'range_y':           axis_stats['y']['range'],
                'abs_err_y':         float(abs_errs[1]),
                'rel_err_y':         float(rel_errs[1]),
                'psnr_y':            float(psnrs[1]),
                'nrmse_y':           float(nrmse),
                'min_z':             axis_stats['z']['min'],
                'max_z':             axis_stats['z']['max'],
                'range_z':           axis_stats['z']['range'],
                'abs_err_z':         float(abs_errs[2]),
                'rel_err_z':         float(rel_errs[2]),
                'psnr_z':            float(psnrs[2]),
                'nrmse_z':           float(nrmse),
                'num_points':        N
            })

        # cleanup flats        
        # remove the flat dirs if (and only if) they're empty
        for d in (tmp / 'flat_compressed', tmp / 'flat_decompressed'):
            try:
                d.rmdir()
            except OSError:
                # if it's not empty (or already gone), just move on
                print("Noticed flat dirs are not empty so not deleting")
                pass

    # finally write CSV
    df = pd.DataFrame(records)
    csv_path = out_root / 'lcp_benchmark.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


if __name__ == '__main__':
    main()
