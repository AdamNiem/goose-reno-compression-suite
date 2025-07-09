
python restore_intensity_feature_dataset_parallel.py \
  --ply_root goose-dataset/reno_decompressed_lidar/Q_512 \
  --orig_bin_root goose-dataset/lidar \
  --out_bin_root goose-dataset/reno_bin_decompressed_lidar/Q_512
  
python restore_intensity_feature_dataset_parallel.py \
  --ply_root goose-dataset/reno_decompressed_lidar/Q_64 \
  --orig_bin_root goose-dataset/lidar \
  --out_bin_root goose-dataset/reno_bin_decompressed_lidar/Q_64
  
python restore_intensity_feature_dataset_parallel.py \
  --ply_root goose-dataset/reno_decompressed_lidar/Q_8 \
  --orig_bin_root goose-dataset/lidar \
  --out_bin_root goose-dataset/reno_bin_decompressed_lidar/Q_8