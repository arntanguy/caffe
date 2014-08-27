#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=../../build/examples/siamese
DATA=../../data/mnist

echo "Creating leveldb..."

rm -rf mnist-siamese-train-leveldb
rm -rf mnist-siamese-test-leveldb

$EXAMPLES/convert_mnist_siamese_data.bin \
    $DATA/train-images-idx3-ubyte \
    $DATA/train-labels-idx1-ubyte \
    mnist-siamese-train-leveldb
$EXAMPLES/convert_mnist_siamese_data.bin \
    $DATA/t10k-images-idx3-ubyte \
    $DATA/t10k-labels-idx1-ubyte \
    mnist-siamese-test-leveldb

echo "Done."
