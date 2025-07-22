# This script takes the outputted folder from the tmc13_compress_goose_dataset.py script 
# and structures it back into a format
# fit for training the ptv3 and lsk3dnet model on
# WARNING: I havent run this script personally since I used the commands myself so maybe inspect before running

# Starting directory for me was 
cd /scratch/aniemcz/goose-pointcept/

cp -r tmc13_compression_results/ tmc13_decompressed_lidar
cd tmc13_decompressed_lidar

mkdir -p Q_0.0001/train
mkdir -p Q_0.0001/trainEx
mkdir -p Q_0.0001/val
mkdir -p Q_0.0001/valEx

mkdir -p Q_0.00521332/train
mkdir -p Q_0.00521332/trainEx
mkdir -p Q_0.00521332/val
mkdir -p Q_0.00521332/valEx

mkdir -p Q_0.0668/train
mkdir -p Q_0.0668/trainEx
mkdir -p Q_0.0668/val
mkdir -p Q_0.0668/valEx

mv train/Q_0.0001/decompressed/* Q_0.0001/train
mv train/Q_0.00521332/decompressed/* Q_0.00521332/train
mv train/Q_0.0668/decompressed/* Q_0.0668/train
mv trainEx/Q_0.0001/decompressed/* Q_0.0001/trainEx
mv trainEx/Q_0.00521332/decompressed/* Q_0.00521332/trainEx
mv trainEx/Q_0.0668/decompressed/* Q_0.0668/trainEx
mv val/Q_0.0001/decompressed/* Q_0.0001/val
mv val/Q_0.00521332/decompressed/* Q_0.00521332/val
mv val/Q_0.0668/decompressed/* Q_0.0668/val
mv valEx/Q_0.0001/decompressed/* Q_0.0001/valEx
mv valEx/Q_0.00521332/decompressed/* Q_0.00521332/valEx
mv valEx/Q_0.0668/decompressed/* Q_0.0668/valEx
rm -r valEx trainEx train val


#sanity check that its the same number of files as original dataset
echo "Number of lidar data files in Goose-Pointcept Dataset Originally"
echo "#####################################################"
cd /scratch/aniemcz/goose-pointcept/
echo "lidar/train: $(find lidar/train -type f | wc -l)"
echo "lidar/trainEx: $(find lidar/trainEx -type f | wc -l)"

echo "lidar/val: $(find lidar/val -type f | wc -l)"
echo "lidar/valEx: $(find lidar/valEx -type f | wc -l)"

echo "Number of lidar data files in tmc13_decomp_lidar dataset now (should match with above)"
echo "#####################################################"
echo "Q_0.0001:"
echo "-----------------------------------------------------"
echo "tmc13_decompressed_lidar/Q_0.0001/train: $(find tmc13_decompressed_lidar/Q_0.0001/train -type f | wc -l)"
echo "tmc13_decompressed_lidar/Q_0.0001/trainEx: $(find tmc13_decompressed_lidar/Q_0.0001/trainEx -type f | wc -l)"

echo "tmc13_decompressed_lidar/Q_0.0001/val: $(find tmc13_decompressed_lidar/Q_0.0001/val -type f | wc -l)"
echo "tmc13_decompressed_lidar/Q_0.0001/valEx: $(find tmc13_decompressed_lidar/Q_0.0001/valEx -type f | wc -l)"

echo "Q_0.00521332:"
echo "-----------------------------------------------------"
echo "tmc13_decompressed_lidar/Q_0.00521332/train: $(find tmc13_decompressed_lidar/Q_0.00521332/train -type f | wc -l)"
echo "tmc13_decompressed_lidar/Q_0.00521332/trainEx: $(find tmc13_decompressed_lidar/Q_0.00521332/trainEx -type f | wc -l)"

echo "tmc13_decompressed_lidar/Q_0.00521332/val: $(find tmc13_decompressed_lidar/Q_0.00521332/val -type f | wc -l)"
echo "tmc13_decompressed_lidar/Q_0.00521332/valEx: $(find tmc13_decompressed_lidar/Q_0.00521332/valEx -type f | wc -l)"

echo "Q_0.0668:"
echo "-----------------------------------------------------"
echo "tmc13_decompressed_lidar/Q_0.0668/train: $(find tmc13_decompressed_lidar/Q_0.0668/train -type f | wc -l)"
echo "tmc13_decompressed_lidar/Q_0.0668/trainEx: $(find tmc13_decompressed_lidar/Q_0.0668/trainEx -type f | wc -l)"

echo "tmc13_decompressed_lidar/Q_0.0668/val: $(find tmc13_decompressed_lidar/Q_0.0668/val -type f | wc -l)"
echo "tmc13_decompressed_lidar/Q_0.0668/valEx: $(find tmc13_decompressed_lidar/Q_0.0668/valEx -type f | wc -l)"


#Now we create multiple folders which will be for each decompressed quantization level 
#This is so that we can have the exact same dataset structure for each quantization level

# Set base source directory
SOURCE_DIR="/scratch/aniemcz/goose-pointcept"
OUTPUT_DIR="/scratch/aniemcz/goose-pointcept-decomp-bin"
tmc13_LIDAR_DIR="${SOURCE_DIR}/tmc13_bin_dequantized_decompressed_lidar"
tmc13_LABEL_DIR="${SOURCE_DIR}/tmc13_bin_dequantized_decompressed_labels_challenge"
COMPRESSOR="tmc13"

mkdir -p "${OUTPUT_DIR}"

# Loop over all Q_* subfolders
for QUALITY_DIR in "${tmc13_LIDAR_DIR}"/Q_*; do
    QUALITY=$(basename "$QUALITY_DIR")
    TARGET_DIR="${OUTPUT_DIR}/${COMPRESSOR}/${QUALITY}"
    LABEL_SOURCE="${tmc13_LABEL_DIR}/${QUALITY}"

    echo "Creating $TARGET_DIR"

    # Create the target directory
    mkdir -p "$TARGET_DIR"

    # Link per-quality labels
    ln -sfn "$LABEL_SOURCE" "$TARGET_DIR/labels_challenge"

    # Link lidar to the appropriate decompressed folder
    ln -sfn "$QUALITY_DIR" "$TARGET_DIR/lidar"
done
