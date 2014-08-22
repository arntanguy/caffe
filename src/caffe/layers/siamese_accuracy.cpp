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
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "Both input images should have the same blob configuration!";
  // Checks if label blob has the correct size
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
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
  //  Dtype accuracy = 0;
  const Dtype* feat1_blob = bottom[0]->cpu_data();
  const Dtype* feat2_blob = bottom[1]->cpu_data();
  const Dtype* label = bottom[2]->cpu_data();

  const int num = bottom[0]->num();
  const int count = bottom[0]->count();
  const int feature_size = bottom[0]->channels();

  Dtype *diff = new Dtype[count];
  caffe_sub(count, feat1_blob, feat2_blob, diff);

  // [0] : average distance for negative loops
  // [1] : average distance for positive loops
  Dtype average_distance[2];
  int num_pos_neg[2];
  int average_count[2] = {1, 1};

  for (int i = 0; i < num; ++i) {
    const Dtype* feat1 = feat1_blob + feature_size*i;
    const Dtype* feat2 = feat2_blob + feature_size*i;
    const int current_label = label[i];
    CHECK(current_label == 0 || current_label == 1) << "Label must be 0 or 1";

    Dtype dot = caffe_cpu_dot(feature_size, feat1, feat2);

    // Rolling average by using a weighted sum
    // Multiply average by current count, add the new element, divide by count+1
    average_distance[current_label] =
        (average_distance[current_label] * average_count[current_label] + dot)
        / (average_count[current_label]+1);
    average_count[current_label]++;
    num_pos_neg[current_label]++;
    //  LOG(INFO) << "Euclidian distance = " << dot << " => "
    //          << ((label[i]==1)?"positive":"negative");

    feat1 += feature_size;
    feat2 += feature_size;
  }

  Dtype margin = average_distance[1]-average_distance[0];
  //  LOG(INFO) << "Average margin between positive and negative: " << margin;

  // Minimal margin
  Dtype epsilon = 0.01;
  if (margin > epsilon) {
    (*top)[0]->mutable_cpu_data()[0] = 1/num;  //  accuracy / num;
  } else {
    (*top)[0]->mutable_cpu_data()[0] = 0;  //  accuracy / num;
  }
}

INSTANTIATE_CLASS(SiameseAccuracyLayer);

}  // namespace caffe
