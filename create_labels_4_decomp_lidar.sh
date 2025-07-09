Q_lvl="Q_8"
python create_labels_4_decompressed_lidar.py \
  --decomp_bin_root /scratch/aniemcz/goose-pointcept/reno_bin_decompressed_lidar/${Q_lvl} \
  --orig_bin_root /scratch/aniemcz/goose-pointcept/lidar \
  --orig_label_root /scratch/aniemcz/goose-pointcept/labels_challenge \
  --out_label_root /scratch/aniemcz/goose-pointcept/reno_bin_decompressed_labels_challenge/${Q_lvl} \
  --threshold 0.007
  
Q_lvl="Q_64"
python create_labels_4_decompressed_lidar.py \
  --decomp_bin_root /scratch/aniemcz/goose-pointcept/reno_bin_decompressed_lidar/${Q_lvl} \
  --orig_bin_root /scratch/aniemcz/goose-pointcept/lidar \
  --orig_label_root /scratch/aniemcz/goose-pointcept/labels_challenge \
  --out_label_root /scratch/aniemcz/goose-pointcept/reno_bin_decompressed_labels_challenge/${Q_lvl} \
  --threshold 0.059
  
Q_lvl="Q_512"
python create_labels_4_decompressed_lidar.py \
  --decomp_bin_root /scratch/aniemcz/goose-pointcept/reno_bin_decompressed_lidar/${Q_lvl} \
  --orig_bin_root /scratch/aniemcz/goose-pointcept/lidar \
  --orig_label_root /scratch/aniemcz/goose-pointcept/labels_challenge \
  --out_label_root /scratch/aniemcz/goose-pointcept/reno_bin_decompressed_labels_challenge/${Q_lvl} \
  --threshold 0.45