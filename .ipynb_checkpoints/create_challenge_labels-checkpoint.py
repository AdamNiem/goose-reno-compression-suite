import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

"""
Parallel semantic label remapping for Goose dataset:
- Reads original .label files (uint32 semantics & instance)
- Extracts semantic labels (lower 16 bits) and instances (upper 16 bits)
- Maps semantic labels from 0–63 to challenge_category_id per provided mapping
- Packs back as (instance<<16) | new_semantic
- Writes out new .label files preserving directory structure

Usage:
python create_challenge_labels.py \
  --input_root ./goose-pointcept/labels \
  --output_root ./goose-pointcept/labels_mapped \
  --num_workers 8
  
python create_challenge_labels.py \
  --input_root /scratch/aniemcz/gooseChallengeLabelsClean/ \
  --output_root /scratch/aniemcz/gooseChallengeLabelsCleanMapped
"""

# Original 64-class -> challenge_category_id mapping
SEM_MAP = {
    0: 0,   # undefined->other
    1: 4,   # traffic_cone->obstacle
    2: 3,   # snow->natural_ground
    3: 2,   # cobble->artificial_ground
    4: 4,   # obstacle->obstacle
    5: 3,   # leaves->natural_ground
    6: 4,   # street_light->obstacle
    7: 2,   # bikeway->artificial_ground
    8: 0,   # ego_vehicle->other
    9: 2,   # pedestrian_crossing->artificial_ground
    10:4,   # road_block->obstacle
    11:2,   # road_marking->artificial_ground
    12:5,   # car->vehicle
    13:5,   # bicycle->vehicle
    14:7,   # person->human
    15:5,   # bus->vehicle
    16:6,   # forest->vegetation
    17:6,   # bush->vegetation
    18:6,   # moss->vegetation
    19:4,   # traffic_light->obstacle
    20:5,   # motorcycle->vehicle
    21:2,   # sidewalk->artificial_ground
    22:2,   # curb->artificial_ground
    23:2,   # asphalt->artificial_ground
    24:3,   # gravel->natural_ground
    25:4,   # boom_barrier->obstacle
    26:4,   # rail_track->obstacle
    27:6,   # tree_crown->vegetation
    28:6,   # tree_trunk->vegetation
    29:4,   # debris->obstacle
    30:6,   # crops->vegetation
    31:3,   # soil->natural_ground
    32:7,   # rider->human
    33:4,   # animal->obstacle
    34:5,   # truck->vehicle
    35:5,   # on_rails->vehicle
    36:5,   # caravan->vehicle
    37:5,   # trailer->vehicle
    38:1,   # building->artificial_structures
    39:1,   # wall->artificial_structures
    40:4,   # rock->obstacle
    41:4,   # fence->obstacle
    42:4,   # guard_rail->obstacle
    43:1,   # bridge->artificial_structures
    44:1,   # tunnel->artificial_structures
    45:4,   # pole->obstacle
    46:4,   # traffic_sign->obstacle
    47:4,   # misc_sign->obstacle
    48:4,   # barrier_tape->obstacle
    49:5,   # kick_scooter->vehicle
    50:3,   # low_grass->natural_ground
    51:6,   # high_grass->vegetation
    52:6,   # scenery_vegetation->vegetation
    53:8,   # sky->sky
    54:3,   # water->natural_ground
    55:4,   # wire->obstacle
    56:0,   # outlier->other
    57:5,   # heavy_machinery->vehicle
    58:4,   # container->obstacle
    59:6,   # hedge->vegetation
    60:4,   # barrel->obstacle
    61:4,   # pipe->obstacle
    62:6,   # tree_root->vegetation
    63:5    # military_vehicle->vehicle
}


def convert_label_file(args):
    label_path, input_root, output_root = args
    rel = label_path.relative_to(input_root)
    out_path = output_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load 32-bit label, extract sem/inst
    data = np.fromfile(str(label_path), dtype=np.uint32)
    sem = data & 0xFFFF
    inst = data >> 16

    # Map semantics
    # Values outside mapping default to 0
    mapped_sem = np.array([SEM_MAP.get(int(s), 0) for s in sem], dtype=np.uint32)

    # Re-pack instance<<16 | sem
    out_data = (inst << 16) | mapped_sem
    out_data.tofile(str(out_path))
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Remap semantic labels from 64 classes to challenge categories'
    )
    parser.add_argument('--input_root', '-i', type=str, required=True,
                        help='Root directory of original .label files')
    parser.add_argument('--output_root', '-o', type=str, required=True,
                        help='Output directory for mapped labels')
    parser.add_argument('--num_workers', '-n', type=int, default=os.cpu_count(),
                        help='Number of parallel workers')
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Gather all .label files
    label_files = list(input_root.rglob('*.label'))
    print(f"Found {len(label_files)} label files under {input_root}")

    tasks = [(p, input_root, output_root) for p in label_files]
    with Pool(processes=args.num_workers) as pool:
        for out in tqdm(pool.imap(convert_label_file, tasks), total=len(tasks), desc="Remapping labels"):
            pass

    print("Label remapping complete.")

if __name__ == '__main__':
    main()
