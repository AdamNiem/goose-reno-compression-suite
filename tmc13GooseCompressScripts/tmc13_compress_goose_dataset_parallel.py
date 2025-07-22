import os
import re
import subprocess
import argparse
import pandas as pd
import csv
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

BITSTREAM_PATTERN = re.compile(r"positions bitstream size (?P<bsz>\d+) B \((?P<bpp>[0-9.]+) bpp\)")
ENC_TIME_PATTERN  = re.compile(r"positions processing time.*: (?P<etime>[0-9.]+) s")
DEC_TIME_PATTERN  = re.compile(r"Processing time \(wall\): (?P<dtime>[0-9.]+) s")

def run_cmd(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    full = []
    for line in proc.stdout:
        print(line, end='')
        full.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed with exit code {proc.returncode}")
    return ''.join(full)

def mirror_and_move(src_flat: Path, dst_root: Path, files, src_root: Path):
    moved = []
    for f in files:
        rel = f.relative_to(src_root)
        src_file = src_flat / f.name
        dst_file = dst_root / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        src_file.replace(dst_file)
        moved.append((f, dst_file))
    return moved

def worker(args):
    in_ply, data_root, out_root, q, tmc3, cfg_path, input_ext = args

    rel = in_ply.relative_to(data_root)
    rel_str = str(rel)
    tmp         = out_root / f"Q_{q}"
    comp_pres   = tmp / 'compressed'
    decomp_pres = tmp / 'decompressed'

    # ─── 1) COMPRESS ──────────────────────────────────────────────────────────
    comp_stream = comp_pres / rel.with_suffix('.bin')
    comp_stream.parent.mkdir(parents=True, exist_ok=True)
    out_c = run_cmd([
        tmc3,
        '--mode=0',
        f'--config={cfg_path}',
        f'--positionQuantizationScale={q}',
        f'--uncompressedDataPath={in_ply}',
        f'--compressedStreamPath={comp_stream}',
    ])
    m_b = BITSTREAM_PATTERN.search(out_c)
    m_e = ENC_TIME_PATTERN.search(out_c)
    bpp         = float(m_b.group('bpp'))
    encode_time = float(m_e.group('etime'))
    total_files = 1

    # mirror compressed
    # now comp_stream *is* our compressed output
    comp_dst = comp_stream

    # ─── 2) DECOMPRESS ────────────────────────────────────────────────────────
    decomp_ply = decomp_pres / rel.with_suffix('.ply')
    decomp_ply.parent.mkdir(parents=True, exist_ok=True)
    out_d = run_cmd([
        tmc3,
        '--mode=1',
        f'--compressedStreamPath={comp_dst}',
        f'--reconstructedDataPath={decomp_ply}',
    ])
    decode_time     = float(DEC_TIME_PATTERN.search(out_d).group('dtime'))
    total_files_dec = 1

    # no mirror step needed — we wrote directly into decomp_pres

    # ─── 3) STATS & ROW ───────────────────────────────────────────────────────
    orig_bytes   = in_ply.stat().st_size
    comp_bytes   = comp_dst.stat().st_size
    decomp_bytes = (decomp_pres/rel.with_suffix('.ply')).stat().st_size
    ratio        = orig_bytes/comp_bytes

    return {
        'rel_path':              rel_str,
        'full_path':             str(in_ply.resolve()),
        'quant':                 q,
        'batch_total_files':     total_files,
        'batch_total_files_dec': total_files_dec,
        'avg_bpp_all':           bpp,
        'encode_time_all_s':     encode_time,
        'decode_time_all_s':     decode_time,
        'orig_bytes':            orig_bytes,
        'comp_bytes':            comp_bytes,
        'decomp_bytes':          decomp_bytes,
        'ratio':                 ratio
    }

def main():
    parser = argparse.ArgumentParser(
        description="Batch-benchmark TMC13 on the Goose dataset, preserving folder layout."
    )
    parser.add_argument('--data_root',      type=str, required=True)
    parser.add_argument('--output_root',    type=str, default='./analysis')
    parser.add_argument('--quant_levels',   nargs='+', type=float, default=[])
    parser.add_argument('--input_file_type',type=str, default="ply", choices=['bin','ply'])
    parser.add_argument('--workers',        type=int, default=4)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root  = Path(args.output_root)
    tmc3      = "/home/aniemcz/rellis/compressionTools/TMC13_compressor/mpeg-pcc-tmc13/mpeg-pcc-tmc13/build/tmc3/tmc3"
    cfg_path  = "./gpcc.cfg"
    input_ext = args.input_file_type.lstrip('.')

    # ─── prepare checkpoint CSV ─────────────────────────────────────────────────
    csv_path = out_root/'compression_benchmark.csv'
    if csv_path.exists():
        df_done = pd.read_csv(csv_path)
        done = set(zip(df_done['quant'], df_done['rel_path']))
        mode = 'a'
        write_header = False
    else:
        done = set()
        out_root.mkdir(parents=True, exist_ok=True)
        # write empty with header
        pd.DataFrame([], columns=[
            'rel_path','full_path','quant',
            'batch_total_files','batch_total_files_dec',
            'avg_bpp_all','encode_time_all_s','decode_time_all_s',
            'orig_bytes','comp_bytes','decomp_bytes','ratio'
        ]).to_csv(csv_path,index=False)
        mode = 'a'
        write_header = False

    # ─── collect tasks ──────────────────────────────────────────────────────────
    all_inputs = sorted(data_root.rglob(f'*.{input_ext}'))
    tasks = []
    for q in args.quant_levels:
        for in_ply in all_inputs:
            rel = str(in_ply.relative_to(data_root))
            if (q, rel) not in done:
                tasks.append((in_ply, data_root, out_root, q, tmc3, cfg_path, input_ext))

    # ─── fieldnames & writer ───────────────────────────────────────────────────
    fieldnames = [
        'rel_path','full_path','quant','batch_total_files','batch_total_files_dec',
        'avg_bpp_all','encode_time_all_s','decode_time_all_s',
        'orig_bytes','comp_bytes','decomp_bytes','ratio'
    ]
    with open(csv_path, mode, newline='') as f, Pool(args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f.flush()

        for row in tqdm(pool.imap_unordered(worker, tasks),
                        total=len(tasks), desc="Overall"):
            if row is not None:
                writer.writerow(row)
                f.flush()

    print(f"Done!  Results in {csv_path}")

if __name__ == '__main__':
    main()
