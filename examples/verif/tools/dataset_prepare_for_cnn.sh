#!/bin/bash

echo "This script must be run from the dataset ROOT directory"
WIDTH=47
HEIGHT=55
RANDOM_SHUFFLE=0

# compute_all = 0 -> only compute new directories
# compute_all = 1 -> compute all
COMPUTE_ALL=0
# Number of frames between two keyframes
KEYFRAME_STEP=5
# Minimal number of keyframes separating two images to be considered as a loop-closure. 
# This is used to reduce the number of redundant loop-closures detected
KEYFRAME_DISTANCE=10
ROTATION=5
TRANSLATION=0.5

echo "Preparing dataset for CNN"
rm -rf db


for dir in */
do
    dir=${dir%%/}
    cd $dir 
    echo "Computing loop-closure dataset for $dir"
    create_loop_closure_dataset.bin
    cd ..
done
echo "Merging all loop-closure datasets"
merge_loop_closure_dataset.py
echo "Creating the shuffle list"
lc_create_shuffle_list.py
echo "Packing images into leveldb dataset"
create_imageset_rgbd.bin loop_closures_dataset.txt db db.txt $WIDTH $HEIGHT $RANDOM_SHUFFLE
