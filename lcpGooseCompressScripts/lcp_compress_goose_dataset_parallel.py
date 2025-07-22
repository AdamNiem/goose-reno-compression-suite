import os
import re
import subprocess
import argparse
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

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

def run_cmd(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    out_lines = []
    for line in proc.stdout:
        print(line, end='')
        out_lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed")
    return ''.join(out_lines)

def worker(args):
    x, data_root, ply_root, out_root, q, lcp = args
    rel = x.relative_to(data_root)
    rel_str = str(rel)
    tmp = out_root / f"EB_{q}"
    comp_pres = tmp / 'compressed'
    decomp_pres = tmp / 'decompressed'
    comp_pres.mkdir(parents=True, exist_ok=True)
    decomp_pres.mkdir(parents=True, exist_ok=True)

    # build file paths
    y = x.with_name(x.stem[:-2] + '_y.dat')
    z = x.with_name(x.stem[:-2] + '_z.dat')

    # number of points
    N = np.fromfile(str(x), dtype=np.float32).size

    # outputs for compress+decompress
    out_lcp = tmp/'flat_compressed'/rel.with_suffix('.lcp')
    out_lcp.parent.mkdir(parents=True, exist_ok=True)
    out_x = tmp/'flat_decompressed'/Path(rel_str)
    out_y = tmp/'flat_decompressed'/Path(rel_str.replace('_x.dat','_y.dat'))
    out_z = tmp/'flat_decompressed'/Path(rel_str.replace('_x.dat','_z.dat'))
    out_x.parent.mkdir(parents=True, exist_ok=True)

    # run compressor + decompressor
    cmd = [
        lcp,
        '-i', str(x), str(y), str(z),
        '-z', str(out_lcp),
        '-o', str(out_x), str(out_y), str(out_z),
        '-1', str(N),
        '-eb', str(q), '-bt', '1', '-a'
    ]
    out = run_cmd(cmd)

    # parse metrics
    ratio       = float(RATIO_PATTERN.search(out).group('ratio'))
    encode_time = float(CTIME_PATTERN.search(out).group('ctime'))
    decode_time = float(D_TIME_PATTERN.search(out).group('dtime'))

    axis_stats = {a: {} for a in 'xyz'}
    for m in AXIS_STATS_PATTERN.finditer(out):
        grp = m.group
        axis_stats[grp('axis')] = {
            'min':   float(grp('min')),
            'max':   float(grp('max')),
            'range': float(grp('range'))
        }
    abs_errs = ABS_ERR_PATTERN.findall(out)
    rel_errs = REL_ERR_PATTERN.findall(out)
    psnrs    = PSNR_PATTERN.findall(out)
    nrmse    = NRMSE_PATTERN.search(out).group('nrmse')

    # move compressed
    comp_dst = comp_pres/rel.with_suffix('.lcp')
    comp_dst.parent.mkdir(parents=True, exist_ok=True)
    out_lcp.replace(comp_dst)

    # move decompressed
    for src, suff in ((out_x,'_x.dat'), (out_y,'_y.dat'), (out_z,'_z.dat')):
        # build new relative path by swapping “_x.dat” etc.
        dst_rel = rel_str.replace('_x.dat', suff)
        dst = decomp_pres / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)

    # original ply size
    ply_file      = ply_root /  Path(rel_str.replace('_x.dat', '.ply'))
    orig_bytes_ply = ply_file.stat().st_size

    return {
        'rel_path':        rel_str,
        'eb':              q,
        'compression_ratio': ratio,
        'encode_time_s':   encode_time,
        'decode_time_s':   decode_time,
        'orig_bytes_dat':  x.stat().st_size,
        'orig_bytes_ply':  orig_bytes_ply,
        'comp_bytes':      comp_dst.stat().st_size,
        'min_x':           axis_stats['x']['min'],
        'max_x':           axis_stats['x']['max'],
        'range_x':         axis_stats['x']['range'],
        'abs_err_x':       float(abs_errs[0]),
        'rel_err_x':       float(rel_errs[0]),
        'psnr_x':          float(psnrs[0]),
        'nrmse_x':         float(nrmse),
        'min_y':           axis_stats['y']['min'],
        'max_y':           axis_stats['y']['max'],
        'range_y':         axis_stats['y']['range'],
        'abs_err_y':       float(abs_errs[1]),
        'rel_err_y':       float(rel_errs[1]),
        'psnr_y':          float(psnrs[1]),
        'nrmse_y':         float(nrmse),
        'min_z':           axis_stats['z']['min'],
        'max_z':           axis_stats['z']['max'],
        'range_z':         axis_stats['z']['range'],
        'abs_err_z':       float(abs_errs[2]),
        'rel_err_z':       float(rel_errs[2]),
        'psnr_z':          float(psnrs[2]),
        'nrmse_z':         float(nrmse),
        'num_points':      N
    }

def main():
    parser = argparse.ArgumentParser(description="Parallel LCP+checkpoint")
    parser.add_argument('--data_root',   required=True)
    parser.add_argument('--ply_root',    required=True)
    parser.add_argument('--output_root', default='./analysis')
    parser.add_argument('--quant_levels', nargs='+', type=float, default=[0.689, 0.2364, 0.085901831, 1e-1, 1e-2, ])
    parser.add_argument('--workers',     type=int,   default=4)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ply_root  = Path(args.ply_root)
    out_root  = Path(args.output_root)
    lcp       = "/home/aniemcz/rellis/compressionTools/lcp_compressor/LCP/compiledExecutable/bin/lcp"

    # build or read checkpoint CSV
    csv_path = out_root/'lcp_benchmark.csv'
    if csv_path.exists():
        df_done = pd.read_csv(csv_path)
        done = set(zip(df_done['eb'], df_done['rel_path']))
        mode = 'a'
        write_header = False
    else:
        done = set()
        out_root.mkdir(parents=True, exist_ok=True)
        mode = 'w'
        write_header = True

    # collect tasks, skipping done
    x_files = sorted(data_root.rglob('*_x.dat'))
    tasks = []
    for q in args.quant_levels:
        for x in x_files:
            rel = str(x.relative_to(data_root))
            if (q, rel) not in done:
                tasks.append((x, data_root, ply_root, out_root, q, lcp))

    # prepare CSV writer
    fieldnames = [
      'rel_path','eb','compression_ratio','encode_time_s','decode_time_s',
      'orig_bytes_dat','orig_bytes_ply','comp_bytes',
      'min_x','max_x','range_x','abs_err_x','rel_err_x','psnr_x','nrmse_x',
      'min_y','max_y','range_y','abs_err_y','rel_err_y','psnr_y','nrmse_y',
      'min_z','max_z','range_z','abs_err_z','rel_err_z','psnr_z','nrmse_z',
      'num_points'
    ]

    with open(csv_path, mode, newline='') as f, Pool(args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f.flush()

        for row in tqdm(pool.imap_unordered(worker, tasks),
                        total=len(tasks), desc="Overall"):
            writer.writerow(row)
            f.flush()

    print("Done — results in", csv_path)

if __name__ == '__main__':
    main()
