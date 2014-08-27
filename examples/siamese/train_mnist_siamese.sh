#!/usr/bin/env sh

TOOLS=../../build/tools

$TOOLS/caffe train --solver=mnist_siamese_solver.prototxt
