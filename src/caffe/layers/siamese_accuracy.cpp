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
  // Genuine average
  (*top)[2]->Reshape(1, 1, 1, 1);
  // Impostor average
  (*top)[3]->Reshape(1, 1, 1, 1);

  correct_genuine = 0;
  incorrect_genuine = 0;
  threshold = 0;

  average_distance_genuine_ra = 0;
  average_distance_impostor_ra = 0;
  number_genuine_ra = 0;
  number_impostor_ra = 0;
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
  const Dtype* distance = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();

  const int num = bottom[0]->num();

  int correct_genuine_batch = 0;
  int incorrect_genuine_batch = 0;

  for (int i = 0; i < num; ++i) {
    const Dtype d = distance[i];
    const int l = label[i];
    //  LOG(INFO) << "Label: " << label[i] << ", Distance: " << d;
    if (l == 0) {
      // Compute running average of genuine distances
      average_distance_genuine_ra =
          (average_distance_genuine_ra * number_genuine_ra + d)
          / (number_genuine_ra+1);
      ++number_genuine_ra;

      if (threshold != 0) {
        if (d < threshold) {
          correct_genuine_batch++;
        } else {
          incorrect_genuine_batch++;
        }
      }
    } else {
      // Compute running average of impostor distances
      average_distance_impostor_ra =
          (average_distance_impostor_ra * number_impostor_ra + d)
          / (number_impostor_ra+1);
      ++number_impostor_ra;

      if (threshold != 0) {
        if (d < threshold) {
          incorrect_genuine_batch++;
        } else {
          correct_genuine_batch++;
        }
      }
    }
  }
  correct_genuine += correct_genuine_batch;
  incorrect_genuine += incorrect_genuine_batch;


  // if (incorrect_genuine_batch != 0) {
  //   LOG(INFO) << "Batch accuracy: "
  //             << correct_genuine_batch /
  //       static_cast<Dtype>(correct_genuine_batch+incorrect_genuine_batch);
  // }

  Dtype accuracy = correct_genuine /
      static_cast<Dtype>(correct_genuine+incorrect_genuine);
  if (threshold == 0) accuracy = 0;

  // LOG(INFO) << "Average distance genuine: "
  //           << average_distance_genuine_ra
  //           << ", impostors: "
  //           << average_distance_impostor_ra;

  threshold = std::min(average_distance_genuine_ra,
                       average_distance_impostor_ra) +
      fabs(average_distance_genuine_ra-average_distance_impostor_ra)/2;
  // LOG(INFO) << "Best threshod for current average distances: " << threshold;

  (*top)[0]->mutable_cpu_data()[0] = accuracy;
  (*top)[1]->mutable_cpu_data()[0] = threshold;
  (*top)[2]->mutable_cpu_data()[0] = average_distance_genuine_ra;
  (*top)[3]->mutable_cpu_data()[0] = average_distance_impostor_ra;
}

INSTANTIATE_CLASS(SiameseAccuracyLayer);

}  // namespace caffe
