# This script takes the outputted folder from the reno_compress_goose_dataset.py script 
# and structures it back into a format
# fit for training the ptv3 and lsk3dnet model on
# WARNING: I havent run this script personally since I used the commands myself so maybe inspect before running

# Starting directory for me was 
cd /scratch/aniemcz/goose-pointcept/

cp -r reno_compression_results/ reno_decompressed_lidar
cd reno_decompressed_lidar

mkdir -p Q_512/train
mkdir -p Q_512/trainEx
mkdir -p Q_512/val
mkdir -p Q_512/valEx

mkdir -p Q_64/train
mkdir -p Q_64/trainEx
mkdir -p Q_64/val
mkdir -p Q_64/valEx

mkdir -p Q_8/train
mkdir -p Q_8/trainEx
mkdir -p Q_8/val
mkdir -p Q_8/valEx

mv train/Q_512/decompressed/* Q_512/train
mv train/Q_64/decompressed/* Q_64/train
mv train/Q_8/decompressed/* Q_8/train
mv trainEx/Q_512/decompressed/* Q_512/trainEx
mv trainEx/Q_64/decompressed/* Q_64/trainEx
mv trainEx/Q_8/decompressed/* Q_8/trainEx
mv val/Q_512/decompressed/* Q_512/val
mv val/Q_64/decompressed/* Q_64/val
mv val/Q_8/decompressed/* Q_8/val
mv valEx/Q_512/decompressed/* Q_512/valEx
mv valEx/Q_64/decompressed/* Q_64/valEx
mv valEx/Q_8/decompressed/* Q_8/valEx
rm -r valEx trainEx train val


#sanity check that its the same number of files as original dataset
echo "Number of lidar data files in Goose-Pointcept Dataset Originally"
echo "#####################################################"
cd /scratch/aniemcz/goose-pointcept/
echo "lidar/train: $(find lidar/train -type f | wc -l)"
echo "lidar/trainEx: $(find lidar/trainEx -type f | wc -l)"

echo "lidar/val: $(find lidar/val -type f | wc -l)"
echo "lidar/valEx: $(find lidar/valEx -type f | wc -l)"

echo "Number of lidar data files in Reno_decomp_lidar dataset now (should match with above)"
echo "#####################################################"
echo "Q_512:"
echo "-----------------------------------------------------"
echo "reno_decompressed_lidar/Q_512/train: $(find reno_decompressed_lidar/Q_512/train -type f | wc -l)"
echo "reno_decompressed_lidar/Q_512/trainEx: $(find reno_decompressed_lidar/Q_512/trainEx -type f | wc -l)"

echo "reno_decompressed_lidar/Q_512/val: $(find reno_decompressed_lidar/Q_512/val -type f | wc -l)"
echo "reno_decompressed_lidar/Q_512/valEx: $(find reno_decompressed_lidar/Q_512/valEx -type f | wc -l)"

echo "Q_64:"
echo "-----------------------------------------------------"
echo "reno_decompressed_lidar/Q_64/train: $(find reno_decompressed_lidar/Q_64/train -type f | wc -l)"
echo "reno_decompressed_lidar/Q_64/trainEx: $(find reno_decompressed_lidar/Q_64/trainEx -type f | wc -l)"

echo "reno_decompressed_lidar/Q_64/val: $(find reno_decompressed_lidar/Q_64/val -type f | wc -l)"
echo "reno_decompressed_lidar/Q_64/valEx: $(find reno_decompressed_lidar/Q_64/valEx -type f | wc -l)"

echo "Q_8:"
echo "-----------------------------------------------------"
echo "reno_decompressed_lidar/Q_8/train: $(find reno_decompressed_lidar/Q_8/train -type f | wc -l)"
echo "reno_decompressed_lidar/Q_8/trainEx: $(find reno_decompressed_lidar/Q_8/trainEx -type f | wc -l)"

echo "reno_decompressed_lidar/Q_8/val: $(find reno_decompressed_lidar/Q_8/val -type f | wc -l)"
echo "reno_decompressed_lidar/Q_8/valEx: $(find reno_decompressed_lidar/Q_8/valEx -type f | wc -l)"


#Now we create multiple folders which will be for each decompressed quantization level 
#This is so that we can have the exact same dataset structure for each quantization level

# Set base source directory
SOURCE_DIR="/scratch/aniemcz/goose-pointcept"
OUTPUT_DIR="/scratch/aniemcz/goose-pointcept-decomp-bin"
RENO_LIDAR_DIR="${SOURCE_DIR}/reno_bin_decompressed_lidar"
RENO_LABEL_DIR="${SOURCE_DIR}/reno_bin_decompressed_labels_challenge"
COMPRESSOR="reno"

mkdir -p "${OUTPUT_DIR}"

# Loop over all Q_* subfolders
for QUALITY_DIR in "${RENO_LIDAR_DIR}"/Q_*; do
    QUALITY=$(basename "$QUALITY_DIR")
    TARGET_DIR="${OUTPUT_DIR}/${COMPRESSOR}/${QUALITY}"
    LABEL_SOURCE="${RENO_LABEL_DIR}/${QUALITY}"

    echo "Creating $TARGET_DIR"

    # Create the target directory
    mkdir -p "$TARGET_DIR"

    # Link per-quality labels
    ln -sfn "$LABEL_SOURCE" "$TARGET_DIR/labels_challenge"

    # Link lidar to the appropriate decompressed folder
    ln -sfn "$QUALITY_DIR" "$TARGET_DIR/lidar"
done
