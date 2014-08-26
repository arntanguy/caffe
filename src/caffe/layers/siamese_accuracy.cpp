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
void SiameseAccuracyLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "SiameseAccuracyLayer takes 2 input blobs:"
                             << "distance (between feature descriptors)"
                             << ", and label";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "Both input descriptors should have the same blob configuration!";
  // Checks if label blob has the correct size
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 1, 1, 1);
}

/**
 * Accuracy = 1 when margin between positive and negative is above a
 * certain threshold
 * XXX: How to properly define threshold?
 **/
template <typename Dtype>
void SiameseAccuracyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  LOG(INFO) << "Accuracy forward";
  const Dtype* distance = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();

  const int num = bottom[0]->num();

  for (int i = 0; i < num; ++i) {
    LOG(INFO) << "Label: " << label[i] << ", Distance: " << distance[i];
  }

  (*top)[0]->mutable_cpu_data()[0] = 1/num;
  // // Minimal margin
  // Dtype epsilon = 0.01;
  // if (margin > epsilon) {
  //   (*top)[0]->mutable_cpu_data()[0] = 1/num;  //  accuracy / num;
  // } else {
  //    (*top)[0]->mutable_cpu_data()[0] = 0;  //  accuracy / num;
  // }
}

INSTANTIATE_CLASS(SiameseAccuracyLayer);

}  // namespace caffe
