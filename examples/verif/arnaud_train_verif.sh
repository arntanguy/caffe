#!/usr/bin/env sh


GLOG_logtostderr=1 train_net.bin arnaud_verif_solver.prototxt \
    dual_extra_param.prototxt $1 2>&1|tee train.log #train_shuffle2.txt

