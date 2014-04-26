#!/usr/bin/env sh

TOOLS=../../build/examples/verif

GLOG_logtostderr=1 $TOOLS/train_net.bin verif_solver.prototxt \
dual_extra_param.prototxt 2>&1|tee train.log #train_shuffle2.txt
