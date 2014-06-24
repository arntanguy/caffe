#!/usr/bin/env sh

TOOLS=../../build/examples/verif

if [ "$1" == "" ];
then
    echo "Resuming training from $1"
fi 

GLOG_logtostderr=1 $TOOLS/train_net.bin arnaud_verif_solver.prototxt \
    dual_extra_param.prototxt $1 2>&1|tee train.log #train_shuffle2.txt

