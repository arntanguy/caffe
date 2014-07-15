#!/bin/bash

echo "This script must be run from the dataset ROOT directory"
WIDTH=47
HEIGHT=55
RANDOM_SHUFFLE=0

cd $(pwd)
for i in $(ls -d */); 
do 
    echo "Associating files in folder ${i}"
    cd ${i%%/}
    associate_multi.py groundtruth.txt rgb.txt depth.txt > associated_all.txt
    dataset_create_from_associated_all_file.py
    rm -rf db
    create_imageset_rgbd.bin dataset.txt db db.txt $WIDTH $HEIGHT $RANDOM_SHUFFLE
    cd ..
done
