// Copyright 2014 BVLC and contributors

#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
void FvHingeLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  LOG(INFO) << "Forward GPU";
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void FvHingeLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_CLASS(FvHingeLossLayer);


}  // namespace caffe
