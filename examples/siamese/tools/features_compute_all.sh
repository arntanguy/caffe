#!/bin/bash

echo "This script must be run from the dataset ROOT directory"

#NET_TRAINED="/home/arnaud/Internship/code/caffe-veri/examples/verif/verif_iter_4000"
NET_TRAINED="/home/arnaud/Internship/code/caffe-veri/examples/siamese/net_siamese_imagenet/siamese_imagenet_trained_9000"
NET_PROTO="/home/arnaud/Internship/code/caffe-veri/examples/siamese/net_siamese_imagenet/imagenet_features.prototxt"

#NET_TRAINED="/home/arnaud/Internship/code/caffe-veri/examples/siamese/net_siamese_imagenet/reference_imagenet_trained"
#NET_PROTO="/home/arnaud/Internship/code/caffe-veri/examples/siamese/net_siamese_imagenet/imagenet_features.prototxt"
BLOB_NAME="fc7"

FEATURES="features_siamese"
FEATURES_DISTANCE="features_distance_siamese_9000.txt"

DATASET="dataset.txt"
DB="db"

rm -rf $FEATURES
extract_features.bin $NET_TRAINED $NET_PROTO $DB $BLOB_NAME $FEATURES GPU
feature_distance.bin $FEATURES $DATASET $FEATURES_DISTANCE 

#cd $(pwd)
#for i in */; 
#do 
#    echo "Extracting features in folder ${i}"
#    cd ${i%%/}
#    rm -rf "$FEATURES"
#    extract_features.bin $NET_TRAINED $NET_PROTO $DB $BLOB_NAME $FEATURES GPU
#    feature_distance.bin $FEATURES $DATASET $FEATURES_DISTANCE 
#    cd ..
#done