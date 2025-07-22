import os
import re
import subprocess
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

#  Example Usage:
#  Note: The goose dataset is large so I instead ran this separately for each subdirectory
#  (it will append csv each time instead of overwriting)

BITSTREAM_PATTERN = re.compile(r"positions bitstream size (?P<bsz>\d+) B \((?P<bpp>[0-9.]+) bpp\)")
ENC_TIME_PATTERN  = re.compile(r"positions processing time.*: (?P<etime>[0-9.]+) s")
DEC_TIME_PATTERN  = re.compile(r"Processing time \(wall\): (?P<dtime>[0-9.]+) s")

def run_cmd(cmd):
    # Launch the process, merging stderr into stdout
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    full_output = []
    for line in proc.stdout:
        print(line, end='')        # real-time echo
        full_output.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed with exit code {proc.returncode}")
    return ''.join(full_output)

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

def main():
    parser = argparse.ArgumentParser(
        description="Batch-benchmark TMC13 on the Goose dataset, preserving folder layout."
    )
    parser.add_argument('--data_root',      type=str, required=True,
                        help='Root directory of goose .bin files')
    parser.add_argument('--output_root',    type=str, default='./analysis',
                        help='Where to write compressed, decompressed data and CSV')
    parser.add_argument('--quant_levels',   nargs='+', type=float, default=[],
                        help='Quantization levels')
    parser.add_argument('--input_file_type', type=str, default="ply",
                        choices=['bin', 'ply'],
                        help="Input's file type for TMC3 compressor")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root  = Path(args.output_root)
    tmc3      = "/home/aniemcz/rellis/compressionTools/TMC13_compressor/mpeg-pcc-tmc13/mpeg-pcc-tmc13/build/tmc3/tmc3"
    cfg_path  = "./gpcc.cfg"

    # ─── NEW: load existing CSV & build a set of (quant,rel_path) already done ───
    csv_path = out_root / 'compression_benchmark.csv'
    if csv_path.exists():
        df_done = pd.read_csv(csv_path)
        done = set(zip(df_done['quant'], df_done['rel_path']))
    else:
        done = set()
        # ensure header will be written on first append
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([], columns=[
            'rel_path','full_path','quant','batch_total_files','batch_total_files_dec',
            'avg_bpp_all','encode_time_all_s','decode_time_all_s',
            'orig_bytes','comp_bytes','decomp_bytes','ratio'
        ]).to_csv(csv_path, index=False)
    # ───────────────────────────────────────────────────────────────────────────────

    input_file_ext = args.input_file_type.lstrip('.')
    all_inputs     = list(data_root.rglob(f'*.{input_file_ext}'))
    if not all_inputs:
        raise Exception(f"No files of type {input_file_ext} at {data_root}")
    if input_file_ext == 'ply':
        all_inputs_bin = [p.with_suffix('.bin') for p in all_inputs]
    else:
        all_inputs_bin = all_inputs

    for q in args.quant_levels:
        print(f"\n=== Quantization: {q} ===")
        temp_root   = out_root / f"Q_{q}"
        comp_flat   = temp_root / 'flat_compressed'
        decomp_flat = temp_root / 'flat_decompressed'
        comp_pres   = temp_root / 'compressed'
        decomp_pres = temp_root / 'decompressed'
        comp_flat.mkdir(parents=True, exist_ok=True)
        decomp_flat.mkdir(parents=True, exist_ok=True)

        # ─── 1) COMPRESS WITH TMC3 ───────────────────────────────────────────────
        for in_ply in tqdm(all_inputs, desc=f"Compress Q={q}"):
            rel = in_ply.relative_to(data_root)
            # ── SKIP if already recorded ──
            if (q, str(rel)) in done:
                continue

            comp_stream = comp_flat / rel.with_suffix('.bin')
            comp_stream.parent.mkdir(parents=True, exist_ok=True)
            out_c = run_cmd([
                tmc3,
                '--mode=0',
                f'--config={cfg_path}',
                f'--positionQuantizationScale={q}',
                f'--uncompressedDataPath={in_ply}',
                f'--compressedStreamPath={comp_stream}',
            ])
            m_bpp   = BITSTREAM_PATTERN.search(out_c)
            m_etime = ENC_TIME_PATTERN.search(out_c)
            bpp         = float(m_bpp.group('bpp'))
            encode_time = float(m_etime.group('etime'))
            total_files = int(m_etime is not None)  # always 1

            moved_comp = mirror_and_move(comp_flat, comp_pres,
                                         all_inputs_bin, data_root)

            # ─── 2) DECOMPRESS WITH TMC3 ─────────────────────────────────────────
            _, comp_dst = moved_comp[-1]
            decomp_ply  = decomp_flat / rel.with_suffix('.ply')
            decomp_ply.parent.mkdir(parents=True, exist_ok=True)
            out_d = run_cmd([
                tmc3,
                '--mode=1',
                f'--compressedStreamPath={comp_dst}',
                f'--reconstructedDataPath={decomp_ply}',
            ])
            decode_time     = float(DEC_TIME_PATTERN.search(out_d).group('dtime'))
            batch_total_dec = 1

            moved_decomp = mirror_and_move(decomp_flat, decomp_pres,
                                           [rel.with_suffix('.ply')], data_root)

            # ─── compute sizes & ratio ───────────────────────────────────────────
            orig_bytes   = in_ply.stat().st_size
            comp_bytes   = comp_dst.stat().st_size
            decomp_bytes = (decomp_pres/rel.with_suffix('.ply')).stat().st_size
            ratio        = orig_bytes/comp_bytes

            row = {
                'rel_path':            str(rel),
                'full_path':           str(in_ply.resolve()),
                'quant':               q,
                'batch_total_files':   total_files,
                'batch_total_files_dec':batch_total_dec,
                'avg_bpp_all':         bpp,
                'encode_time_all_s':   encode_time,
                'decode_time_all_s':   decode_time,
                'orig_bytes':          orig_bytes,
                'comp_bytes':          comp_bytes,
                'decomp_bytes':        decomp_bytes,
                'ratio':               ratio
            }

            # ─── APPEND this one row to CSV ─────────────────────────────────────
            pd.DataFrame([row]).to_csv(csv_path,
                                      mode='a',
                                      header=False,
                                      index=False)
            done.add((q, str(rel)))
            # ────────────────────────────────────────────────────────────────────

        # clean up flats
        for d in (comp_flat, decomp_flat):
            for f in d.rglob('*'):
                f.unlink()
            d.rmdir()

    print(f"Completed. Results in {csv_path}")

if __name__ == '__main__':
    main()
