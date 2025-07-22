lcp="/home/aniemcz/rellis/compressionTools/lcp_compressor/LCP/compiledExecutable/bin/lcp"

x_dat_path="/home/aniemcz/gooseReno/goose-dataset/dat_xyz_only_lidar/train/2023-04-20_campus/2023-04-20_campus__0286_1681996776758328417_vls128_x.dat"

y_dat_path="/home/aniemcz/gooseReno/goose-dataset/dat_xyz_only_lidar/train/2023-04-20_campus/2023-04-20_campus__0286_1681996776758328417_vls128_y.dat"

z_dat_path="/home/aniemcz/gooseReno/goose-dataset/dat_xyz_only_lidar/train/2023-04-20_campus/2023-04-20_campus__0286_1681996776758328417_vls128_z.dat"

num_points=$(pixi run python -c "import numpy as np; print(np.fromfile(\"${x_dat_path}\", dtype=np.float32).shape[0])")

error_bound="0.689"

echo "error_bound is ${error_bound}"

# Run the LCP compressor
${lcp} -i ${x_dat_path} ${y_dat_path} ${z_dat_path} \
-z compressedLidar.lcp \
-o x_dat.out y_dat.out z_dat.out \
-1 ${num_points} \
-eb ${error_bound} \
-bt 1 \
-a

echo "number of points is ${num_points}"

python -c "import os; print( 'compressed size in bytes: ', os.path.getsize('compressedLidar.lcp') )"

python -c "import os; print( 'avg bpp is: ', (8 * os.path.getsize('compressedLidar.lcp')) / ${num_points} )"

python -c "import os; print( 'calculation used for avg bpp: ', '(8 * os.path.getsize(\'compressedLidar.lcp\'))', ' / ${num_points}')"



















#pretend this doesnt exist for now

x_dat_path2="/home/aniemcz/gooseReno/goose-dataset/dat_xyz_only_lidar/train/2022-07-22_flight/2022-07-22_flight__0000_1658492967230070008_vls128_x.dat"

y_dat_path2="/home/aniemcz/gooseReno/goose-dataset/dat_xyz_only_lidar/train/2022-07-22_flight/2022-07-22_flight__0000_1658492967230070008_vls128_y.dat"

z_dat_path2="/home/aniemcz/gooseReno/goose-dataset/dat_xyz_only_lidar/train/2022-07-22_flight/2022-07-22_flight__0000_1658492967230070008_vls128_z.dat"