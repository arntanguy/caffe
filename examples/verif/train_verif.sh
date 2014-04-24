#!/usr/bin/env sh

TOOLS=../../build/examples/verif

GLOG_logtostderr=1 $TOOLS/train_net.bin verif_solver.prototxt shuffle_list.txt
