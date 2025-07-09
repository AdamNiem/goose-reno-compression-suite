# TMC13
tmc3="/home/aniemcz/rellis/compressionTools/TMC13_compressor/mpeg-pcc-tmc13/mpeg-pcc-tmc13/build/tmc3/tmc3"
data_path="/home/aniemcz/gooseReno/goose-data-examples/2023-04-20_campus__0286_1681996776758328417_vls128.ply"
cfg_path="./gpcc.cfg"
cfg2_path="./gpcc2.cfg"
echo "TMC13: XYZ + NORMALS COMPRESSION\n\n"
${tmc3} \
  --mode=0 \
  --config=gpcc.cfg \
  --positionQuantizationScale=0.5 \
  --uncompressedDataPath=./data/ford_vox1mm_examples/Ford_02_vox1mm-0100.ply \
  --compressedStreamPath=./data/Ford_02_vox1mm-0100.ply.bin
  
${tmc3} \
  --mode=1 \
  --compressedStreamPath=./data/Ford_02_vox1mm-0100.ply.bin \
  --reconstructedDataPath=./data/Ford_02_vox1mm-0100.ply.bin.ply
  
./pc_error_d \
  --fileA=./data/ford_vox1mm_examples/Ford_02_vox1mm-0100.ply \
  --fileB=./data/Ford_02_vox1mm-0100.ply.bin.ply \
  --resolution=59.70
  
  # \ # seems to be able to tune bpp if its true or false
  #
  ## --reconstructedDataPath=recongpcc.ply \
  # \ #also tunable
  # #ok now this is tunable for bpp
 # --positionQuantisationOctreeDepth=2 \
 #--bitdepth=20 \
 #   --qp=64 \   
 
 #--config=/home/aniemcz/RENO/third_party/gpcc.cfg


# 2.93394 bpp w/o config file
# (2.66556 bpp) including config file

#--config=${cfg2_path} \


#source activate lidarDataVisualEnv && \
#python3 psnr_ply.py ${data_path} decomp_gcc.ply


# Findings:
# so qp param does nothing it seems

# data_path="/home/aniemcz/gooseReno/goose-data-examples/testFile.ply"