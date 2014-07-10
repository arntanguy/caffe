#!/bin/sh

# Author: Arnaud TANGUY
#
# This script converts the washington dataset to a for close to the TUM SLAM dataset
# in order to make it usable by the other CNN scripts

echo "This script must be ran from the root of Microsoft scene dataset" 
echo "./convert_microsoft_dataset.sh <save_dir>"

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    exit
fi

DEST="$(realpath $1)"

echo "Running from directory $(pwd)"
echo "Creating destination directory $1"
mkdir -p $DEST

echo "Unzipping zip files"
#unzip *.zip
for dir in $(ls -d */);
do
    echo "======================================"
    echo "Processing $dir"
    echo "======================================"
    echo; echo
    cd $dir
    dst_scene_dir="$DEST/$dir"
    echo "dst_scene_dir $dst_scene_dir"
    echo "Unzipping $(ls *.zip)"
    #unzip *.zip
    for seq in $(ls -d */)
    do
        echo; echo;
        echo "===================================="
        echo "Processing sequence $seq"
        echo "===================================="
        cd $seq
        rm Thumbs.db 2&>/dev/null
        size=$(( $(ls | wc -l) / 3 ))
        echo "Size: $size"
        dst_seq_dir="${dst_scene_dir%%/}-$seq"
        dst_seq_dataset="${dst_seq_dir}dataset.txt"
        dst_rgb="${dst_seq_dir}rgb"
        dst_depth="${dst_seq_dir}depth"
        echo "Source: $(pwd) -> Destination: $dst_seq_dir"
        echo "Destination dataset $dst_seq_dataset"
        mkdir -p $dst_rgb
        mkdir -p $dst_depth 
        rm -i $dst_seq_dataset

        for i in $(seq -f "%06g" 0 $(($size-1)))
        do
            src_rgb="frame-$i.color.png"
            src_depth="frame-$i.depth.png"
            src_pose="frame-$i.pose.txt"
            dst_rgb_img="$dst_rgb/$src_rgb"
            dst_depth_img="$dst_depth/$src_depth"
            echo "cp $src_rgb -> $dst_rgb/"
            mv $src_rgb $dst_rgb
            echo "cp $src_depth -> $dst_depth/"
            mv $src_depth $dst_depth
            #cat $src_pose
            translation=$(cat $src_pose | awk '$1=$1' | cut -d' ' -f4 | while read line; do val=$(echo "scale=6;$(echo  "$line" | sed 's/e/\*10\^/' | sed 's/+//')" | bc); echo  -n "$val "; done)
            declare -a r=(0 0 0 0 0 0 0)
            index=0
            rotation_csv=$(cat $src_pose | awk '$1=$1' | cut -d' ' -f1-3 | while read line; do echo -n "$line "; done)

            r=(0 0 0 0 0 0)
            index=0
            for val in $rotation_csv;
            do
                r[$index]=$(echo "scale=6;$(echo "$val" | sed 's/e/\*10\^/' | sed 's/+//')" | bc)
                index=$((index+1))
            done
            #echo "RMAT: ${r[0]} ${r[1]} ${r[2]} ${r[3]} ${r[4]} ${r[5]} ${r[6]} ${r[7]} ${r[8]}"
            #echo "qw: scale=6;sqrt(1 + ${r[0]} + ${r[4]} + ${r[8]})/2"
            #echo "qx: scale=6;(${r[7]}-${r[5]})/(4*$qw)"
            #echo "qy: scale=6;(${r[2]}-${r[6]})/(4*$qw)"
            #echo "qz: scale=6;(${r[3]}-${r[1]})/(4*$qw)"

            # Convert rotation (pure rotation: orthogonal matrix) to quaternion
            #qw= √(1 + m00 + m11 + m22) /2
            #qx = (m21 - m12)/( 4 *qw)
            #qy = (m02 - m20)/( 4 *qw)
            #qz = (m10 - m01)/( 4 *qw)
            qw=$(echo "scale=6;sqrt(1 + ${r[0]} + ${r[4]} + ${r[8]})/2" | bc)
            qx=$(echo "scale=6;(${r[7]}-(${r[5]}))/(4*$qw)" | bc)
            qy=$(echo "scale=6;(${r[2]}-(${r[6]}))/(4*$qw)" | bc)
            qz=$(echo "scale=6;(${r[3]}-(${r[1]}))/(4*$qw)" | bc)
            echo "$i $dst_rgb_img $dst_depth_img $translation$qx $qy $qz $qw" >> $dst_seq_dataset

        done
        cd ..
     done
    cd ..
done

