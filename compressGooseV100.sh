pixi run python reno_compress_goose_dataset.py   --data_root /home/aniemcz/gooseReno/goose-dataset/ply_xyz_only_lidar/val   --ckpt ./RENO/model/Goose/ckpt.pt   --quant_levels 8 64 512   --output_root /home/aniemcz/gooseReno/goose-dataset/reno_decompressed_lidar/val

pixi run python reno_compress_goose_dataset.py   --data_root /home/aniemcz/gooseReno/goose-dataset/ply_xyz_only_lidar/trainEx   --ckpt ./RENO/model/Goose/ckpt.pt   --quant_levels 8 64 512   --output_root /home/aniemcz/gooseReno/goose-dataset/reno_decompressed_lidar/trainEx

pixi run python reno_compress_goose_dataset.py   --data_root /home/aniemcz/gooseReno/goose-dataset/ply_xyz_only_lidar/train   --ckpt ./RENO/model/Goose/ckpt.pt   --quant_levels 8 64 512   --output_root /home/aniemcz/gooseReno/goose-dataset/reno_decompressed_lidar/train
