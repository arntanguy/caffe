#!/bin/bash

echo "This script must be run from the dataset ROOT directory"
echo "In case of the washington dataset, the file dataset.txt has already been created by the conversion script"
WIDTH=47
HEIGHT=55
RANDOM_SHUFFLE=0

cd $(pwd)
for i in $(ls -d */); 
do 
    echo "Processing ${i}"
    cd ${i%%/}
    rm -rf db
    create_imageset_rgbd.bin dataset.txt db db.txt $WIDTH $HEIGHT $RANDOM_SHUFFLE
    cd ..
done
