# TMC13
tmc3="/home/aniemcz/rellis/compressionTools/TMC13_compressor/mpeg-pcc-tmc13/mpeg-pcc-tmc13/build/tmc3/tmc3"
data_path="/home/aniemcz/gooseReno/goose-data-examples/2023-04-20_campus__0286_1681996776758328417_vls128.ply"
cfg_path="/home/aniemcz/rellis/compressionTools/RENO_compressor/RENO/third_party/gpcc.cfg"
cfg2_path="./gpcc2.cfg"
echo "TMC13: XYZ + NORMALS COMPRESSION\n\n"
${tmc3} \
  --mode=0 \
  --uncompressedDataPath=${data_path} \
  --compressedStreamPath=testgpcc.bin \
  --frameCount=1 \
  --geomTreeType=1 \

  #--cfg=${cfg_path}
  
${tmc3} \
  --mode=1 \
  --outputBinaryPly=0 \
  --compressedStreamPath=testgpcc.bin  \
  --reconstructedDataPath=decomp_gcc.ply  \
  





./pc_error_d \
  --fileA=${data_path} \
  --fileB=decomp_gcc.ply \


#  --resolution=59.70 \
  
  # \ # seems to be able to tune bpp if its true or false
  #
  ## --reconstructedDataPath=recongpcc.ply \
  # \ #also tunable
  # #ok now this is tunable for bpp
 # --positionQuantisationOctreeDepth=2 \
 #--bitdepth=20 \
 #   --qp=64 \   
 
 #--config=/home/aniemcz/RENO/third_party/gpcc.cfg
 
 #  --positionQuantisationEnabled=1 \
 # --positionQuantizationScale=1 \
 # --positionBaseQp=2 \ 
 # --positionSliceQpOffset=2


# 2.93394 bpp w/o config file
# (2.66556 bpp) including config file

#--config=${cfg2_path} \


#source activate lidarDataVisualEnv && \
#python3 psnr_ply.py ${data_path} decomp_gcc.ply


# Findings:
# so qp param does nothing it seems

# data_path="/home/aniemcz/gooseReno/goose-data-examples/testFile.ply"