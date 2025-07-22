Q_lvl="Q_0.0668"
python create_labels_4_decompressed_lidar.py \
  --decomp_bin_root /scratch/aniemcz/goose-pointcept/tmc13_bin_dequantized_decompressed_lidar/${Q_lvl} \
  --orig_bin_root /scratch/aniemcz/goose-pointcept/lidar \
  --orig_label_root /scratch/aniemcz/goose-pointcept/labels_challenge \
  --out_label_root /scratch/aniemcz/goose-pointcept/tmc13_bin_dequantized_decompressed_labels_challenge/${Q_lvl} \
  --no_threshold
  
Q_lvl="Q_0.00521332"
python create_labels_4_decompressed_lidar.py \
  --decomp_bin_root /scratch/aniemcz/goose-pointcept/tmc13_bin_dequantized_decompressed_lidar/${Q_lvl} \
  --orig_bin_root /scratch/aniemcz/goose-pointcept/lidar \
  --orig_label_root /scratch/aniemcz/goose-pointcept/labels_challenge \
  --out_label_root /scratch/aniemcz/goose-pointcept/tmc13_bin_dequantized_decompressed_labels_challenge/${Q_lvl} \
  --no_threshold
  
Q_lvl="Q_0.0001"
python create_labels_4_decompressed_lidar.py \
  --decomp_bin_root /scratch/aniemcz/goose-pointcept/tmc13_bin_dequantized_decompressed_lidar/${Q_lvl} \
  --orig_bin_root /scratch/aniemcz/goose-pointcept/lidar \
  --orig_label_root /scratch/aniemcz/goose-pointcept/labels_challenge \
  --out_label_root /scratch/aniemcz/goose-pointcept/tmc13_bin_dequantized_decompressed_labels_challenge/${Q_lvl} \
  --no_threshold