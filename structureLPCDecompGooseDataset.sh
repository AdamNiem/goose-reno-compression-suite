# This script takes the outputted folder from the lcp_compress_goose_dataset.py script 
# and structures it back into a format
# fit for training the ptv3 and lsk3dnet model on
# WARNING: I havent run this script personally since I used the commands myself so maybe inspect before running

# Starting directory for me was 
cd /scratch/aniemcz/goose-pointcept/

cp -r lcp_compression_results lcp_decompressed_lidar
cd lcp_decompressed_lidar

# List of error bounds
error_bounds=(
  "0.689"
  "0.2364"
  "0.1"
  "0.085901831"
  "0.01"
)

# Create directories and move decompressed folders
for eb in "${error_bounds[@]}"; do
  mv EB_${eb}/decompressed/train   EB_${eb}/
  mv EB_${eb}/decompressed/trainEx EB_${eb}/
  mv EB_${eb}/decompressed/val     EB_${eb}/
  mv EB_${eb}/decompressed/valEx   EB_${eb}/
  
  rm -r EB_${eb}/compressed
  rm -r EB_${eb}/flat_compressed
  rm -r EB_${eb}/flat_decompressed
  rm -r EB_${eb}/decompressed
done

# Sanity check that it's the same number of files as original dataset
echo "Number of lidar data files in Goose-Pointcept Dataset Originally"
echo "#####################################################"
cd /scratch/aniemcz/goose-pointcept/
echo "lidar/train:   $(find lidar/train -type f | wc -l)"
echo "lidar/trainEx: $(find lidar/trainEx -type f | wc -l)"
echo "lidar/val:     $(find lidar/val -type f | wc -l)"
echo "lidar/valEx:   $(find lidar/valEx -type f | wc -l)"
echo ""

# Now check decompressed sets
echo "Number of lidar data files in lcp_decomp_lidar dataset now (NOTE: should be 3x of number above)"
echo "#####################################################"

error_bounds=(
  "0.689"
  "0.2364"
  "0.1"
  "0.085901831"
  "0.01"
)

for eb in "${error_bounds[@]}"; do
  echo "EB_${eb}:"
  echo "-----------------------------------------------------"
  for subset in train trainEx val valEx; do
    count=$(find "lcp_decompressed_lidar/EB_${eb}/${subset}" -type f | wc -l)
    echo "lcp_decompressed_lidar/EB_${eb}/${subset}: $count"
  done
  echo ""
done


#Now we create multiple folders which will be for each decompressed quantization level 
#This is so that we can have the exact same dataset structure for each quantization level

# Set base source directory
SOURCE_DIR="/scratch/aniemcz/goose-pointcept"
OUTPUT_DIR="/scratch/aniemcz/goose-pointcept-decomp-bin"
lcp_LIDAR_DIR="${SOURCE_DIR}/lcp_bin_decompressed_lidar"
lcp_LABEL_DIR="${SOURCE_DIR}/lcp_bin_decompressed_labels_challenge"
COMPRESSOR="lcp"

mkdir -p "${OUTPUT_DIR}"

# Loop over all EB_* subfolders
for QUALITY_DIR in "${lcp_LIDAR_DIR}"/EB_*; do
    QUALITY=$(basename "$QUALITY_DIR")
    TARGET_DIR="${OUTPUT_DIR}/${COMPRESSOR}/${QUALITY}"
    LABEL_SOURCE="${lcp_LABEL_DIR}/${QUALITY}"

    echo "Creating $TARGET_DIR"

    # Create the target directory
    mkdir -p "$TARGET_DIR"

    # Link per-quality labels
    ln -sfn "$LABEL_SOURCE" "$TARGET_DIR/labels_challenge"

    # Link lidar to the appropriate decompressed folder
    ln -sfn "$QUALITY_DIR" "$TARGET_DIR/lidar"
done
