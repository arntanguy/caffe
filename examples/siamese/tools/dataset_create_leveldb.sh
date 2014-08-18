#!/bin/bash

echo "This script must be run from the dataset ROOT directory"
echo "The file dataset.txt must exist in all subdirectories"
WIDTH=47
HEIGHT=55
RANDOM_SHUFFLE=0
DB_BACKEND="lmdb"

cd $(pwd)
for i in $(ls -d */); 
do 
    echo "Processing ${i}"
    cd ${i%%/}
    rm -rf db
    create_imageset_rgbd.bin dataset.txt db db.txt $WIDTH $HEIGHT $DB_BACKEND $RANDOM_SHUFFLE
    cd ..
done
