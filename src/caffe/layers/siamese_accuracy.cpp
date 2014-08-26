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
  // Accuracy
  (*top)[0]->Reshape(1, 1, 1, 1);
  // Threshold
  (*top)[1]->Reshape(1, 1, 1, 1);
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

  static int number_genuine = 0;
  static int number_impostor = 0;
  static Dtype distance_genuine = 0;
  static Dtype distance_impostor = 0;
  static Dtype correct_genuine = 0;
  static Dtype incorrect_genuine = 0;
  static Dtype threshold = 0;

  int correct_genuine_batch = 0;
  int incorrect_genuine_batch = 0;
  int number_genuine_batch = 0;
  int number_impostor_batch = 0;
  Dtype distance_genuine_batch = 0;
  Dtype distance_impostor_batch = 0;


  for (int i = 0; i < num; ++i) {
    const Dtype d = fabs(distance[i]);
    const int l = label[i];
    //LOG(INFO) << "Label: " << label[i] << ", Distance: " << d;
    if (l == 1) {
      number_genuine_batch++;
      distance_genuine_batch += d;
      if (threshold != 0) {
        if (d > threshold) {
          correct_genuine_batch++;
        } else {
          incorrect_genuine_batch++;
        }
      }
    } else {
      number_impostor_batch++;
      distance_impostor_batch += d;
      if (threshold != 0) {
        if (d > threshold) {
          incorrect_genuine_batch++;
        } else {
          correct_genuine_batch++;
        }
      }
    }
  }
  number_genuine += number_genuine_batch;
  number_impostor += number_impostor_batch;
  distance_genuine += distance_genuine_batch;
  distance_impostor += distance_impostor_batch;
  correct_genuine += correct_genuine_batch;
  incorrect_genuine += incorrect_genuine_batch;


  LOG(INFO) << "Average distance genuine (batch): "
            << distance_genuine_batch/number_genuine_batch
            << ", impostors: "
            << distance_impostor_batch/number_impostor_batch;
  LOG(INFO) << "Batch accuracy: " << correct_genuine_batch/static_cast<float>(num);
  LOG(INFO) << "Threshold: " << threshold;

  Dtype accuracy = correct_genuine / (correct_genuine+incorrect_genuine);
  if (threshold == 0) accuracy = 0;
  const Dtype average_distance_genuine = distance_genuine/number_genuine;
  const Dtype average_distance_impostor = distance_impostor/number_impostor;
  LOG(INFO) << "Average distance genuine: "
            << distance_genuine/number_genuine
            << ", impostors: "
            << distance_impostor/number_impostor;


  threshold = average_distance_genuine +
      fabs(average_distance_genuine-average_distance_impostor)/2;
  (*top)[0]->mutable_cpu_data()[0] = accuracy;
  (*top)[1]->mutable_cpu_data()[0] = threshold;
}

INSTANTIATE_CLASS(SiameseAccuracyLayer);

}  // namespace caffe
