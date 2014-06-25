#!/bin/sh

# Author: Arnaud TANGUY
#
# This script converts the washington dataset to a format close to the TUM SLAM dataset
# in order to make it usable by the other CNN scripts

echo "This script must be ran from the root of Washington's scene dataset"
echo "./convert_washington_dataset.sh <save_dir>"

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    exit
fi

DEST="$1"

echo "Running from directory $(pwd)"
echo "Creating destination directory $1"
mkdir $DEST


dir_number=1
for src_pose_file in $(ls pc/*.pose)
do
    size=$(cat $src_pose_file | wc -l)
    src_img_dir="imgs/$(ls imgs/ | head -n $dir_number | tail -n 1)"
    dst_dir="$DEST/washington_$dir_number"
    dst_dataset="$dst_dir/dataset.txt"
    dst_rgb="$dst_dir/rgb"
    dst_depth="$dst_dir/depth"
    rm -rf $dst_dir
    mkdir -p $dst_rgb
    mkdir -p $dst_depth

    echo; echo
    echo "=========================================="
    echo " Processing $src_img_dir "
    echo "=========================================="
    echo

    echo "Source imgage dir $src_img_dir"
    echo "Source pose file $src_pose_file with $size elements"
    echo
    echo "Destination dir $dst_dir"
    echo "Destination dataset $dst_dataset"
    echo "Destination rgb $dst_rgb"
    echo "Destination depth $dst_depth"
    line_num=1
    for i in $(seq -f "%05g" 0 $(($size-1)))
    do
        src_rgb="$src_img_dir/$i-color.png"
        src_depth="$src_img_dir/$i-depth.png"
        dst_rgb_img="$dst_rgb/$i-color.png"
        dst_depth_img="$dst_depth/$i-depth.png"

        #echo "Copying $src_rgb to $dst_rgb_img"
        cp $src_rgb $dst_rgb_img
        #echo "Copying $src_depth to $dst_depth_img"
        cp $src_depth $dst_depth
        #echo "Adding to dataset"
        pose_line=$(cat $src_pose_file | head -n $line_num | tail -n 1)
        translation=$(echo $pose_line | cut -d' ' -f5-7)
        rotation=$(echo $pose_line | cut -d' ' -f1-4)
        echo "$i $dst_rgb_img $dst_depth_img $translation $rotation" >> $dst_dataset 
        line_num=$(($line_num+1))
    done
    dir_number=$((dir_number+1))
done
