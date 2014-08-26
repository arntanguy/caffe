// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void FvHingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int num = bottom[0]->num();

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();

  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    loss += max(Dtype(1.) - bottom_data[i] * bottom_label[i], Dtype(0.));
  }
  (*top)[0]->mutable_cpu_data()[0] = loss/num;
}

template <typename Dtype>
void FvHingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  int num = (*bottom)[0]->num();

  // Compute the gradient
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  const Dtype* bottom_label = (*bottom)[1]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();

  for (int i = 0; i < num; i++) {
    if ( Dtype(1.) - bottom_data[i] * bottom_label[i] > 0 ) {
      bottom_diff[i] = - bottom_label[i] / num;
    } else {
      bottom_diff[i] = Dtype(0.);
    }
  }
}

INSTANTIATE_CLASS(FvHingeLossLayer);

}  // namespace caffe
