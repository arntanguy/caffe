#!/bin/bash

echo "This script must be run from the dataset ROOT directory"
WIDTH=47
HEIGHT=55
RANDOM_SHUFFLE=0

#NET_TRAINED="/home/arnaud/Internship/code/caffe-veri/examples/verif/verif_iter_4000"
NET_TRAINED="/home/arnaud/Internship/code/caffe-veri/examples/verif/verif_iter_1000"
NET_PROTO="/home/arnaud/Internship/code/caffe-veri/examples/verif/arnaud_verif_feature.prototxt"
BLOB_NAME="ip2"

FEATURES="features"
FEATURES_DISTANCE="features_distance.txt"

DATASET="dataset.txt"
DB="db"

cd $(pwd)
for i in */; 
do 
    echo "Extracting features in folder ${i}"
    cd ${i%%/}
    rm -rf features
    extract_features.bin $NET_TRAINED $NET_PROTO $DB $BLOB_NAME $FEATURES GPU
    feature_distance.bin $FEATURES $DATASET $FEATURES_DISTANCE 
    cd ..
done
