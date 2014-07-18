// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
Dtype VerificationLossLayer<Dtype>::CalcThreshold(bool update) {
  int i, j, is, id, is_ = 0, id_ = 0;
  Dtype th, th_c, s, d, f;
  int n = same_.size();
  CHECK_EQ(n, distance_.size());
  if(!n)
    return M_;
  for(i = 0; i < n; i++)
  {
    if(same_[i]) {
      is_++;
    } else {
      id_++;
    }
  }

  Dtype stat[3];
  stat[0] = 1.0;
  stat[1] = 0.5;
  stat[2] = 0.5;
  th = -1.0;

  for(i = 0; i < 4000; i++)
  {
    th_c = i * 0.1;
    is = 0;
    id = 0;
    for(j = 0; j < n; j++)
    {
      if(same_[j]) {
        if(distance_[j] > th_c) {
          is++;
        }
      }
      else {
        if(distance_[j] <= th_c) {
          id++;
        }
      }
    }
    s = (Dtype)is / (2 * is_);
    d = (Dtype)id / (2 * id_);
    f = s + d;
    if(f < stat[0]) {
      stat[0] = f;
      stat[1] = s;
      stat[2] = d;
      th = th_c;
    }
  }
  LOG(INFO) << "margin: " << th << " ("
      << stat[0] << ", " << stat[1]
      << ", " << stat[2] << ")";

  if(update)
    SetThreshold(th);
  return th;
}

template <typename Dtype>
void VerificationLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                         vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 3) << "VerificationLoss Layer takes 3 blobs as input: 2 data blobs, 1 label blob.";
  CHECK_EQ(top->size(), 0) << "VerificationLoss Layer takes no blob as output.";

  diffy1_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diffy2_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  //dual_lamda: 0.05
  //dual_threshold: 39.9
  M_ = 39.9;
  LAMDA_ = 0.05;
  //M_ = this->layer_param_.dual_threshold();
  //LAMDA_ = this->layer_param_.dual_lamda();

  ResetDistanceStat();
  LOG(INFO) << "Initial: threshold " << M_ << ", " << "lamda: " << LAMDA_;
}

template <typename Dtype>
Dtype VerificationLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LOG(INFO) << "Forward_cpu not implemented";
  return Dtype(0);
}

template <typename Dtype>
void VerificationLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                const vector<bool>& propagate_down,
                                                vector<Blob<Dtype>*>* bottom) {
  LOG(INFO) << "Backward_cpu not implemented";
}


INSTANTIATE_CLASS(VerificationLossLayer);


}  // namespace caffe
