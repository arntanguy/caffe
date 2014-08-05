// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/util/debug.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void SiameseLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 3) << "SiameseLoss Layer takes three blobs as input: 2 images, 1 label.";
  CHECK_EQ(top->size(), 0) << "SiameseLoss Layer takes no blob as output.";

  LOG(INFO) << "SetUp";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  LOG(INFO) << diff_;
}

template <typename Dtype>
Dtype SiameseLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                vector<Blob<Dtype>*>* top) {
  /**
   * Loss function as defined in the paper
   * "Learning a Similarity Metric Discriminatively, with Application to Face
   * Verification" - LeCun et al.
   *
   * X1, X2: input images
   * Gw(X1): output descriptor for input X1
   * Gw(X2): output descriptor for input X2
   * TODO
   * Ew(X1, X2) = || Gw(X1) - Gw(X2) ||1   : output energy of the network
   * 
   **/

  DLOG(INFO) << "Number of input blobs: " << bottom.size();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int feature_size = bottom[0]->channels();
  DLOG(INFO) << "Ouput blob info: " << *bottom[0];

  const Dtype* Gw_x1 = bottom[0]->cpu_data();
  const Dtype* Gw_x2 = bottom[1]->cpu_data();
  // label
  const Dtype* labels = bottom[2]->cpu_data();

  /**
   * Compute the output energy of the network
   **/
  Dtype *sub_x1_x2 = diff_.mutable_cpu_data(); 
  caffe_sub(count, Gw_x1, Gw_x2, sub_x1_x2);
  
  Dtype *sub = sub_x1_x2;
  // XXX: should be set to the upper bound of Ew
  // Upper bound: (max_feature_descriptor_value-min_feature_descriptor_value) * feature_size
  // Here assumes max=1, min=0
  const Dtype Q = feature_size; 
  LOG(INFO) << "Max energy: " << Q;
  Dtype loss = 0;
  for(int i=0; i< num; i++) {
    const int Y = labels[i];
    // Norm L1 (sum of absolute values)
    Dtype Ew = caffe_cpu_asum(feature_size, sub);
    Dtype L = (1 - Y) * 2/Q * Ew * Ew  // Increase energy for false positives
              + 2*Y*Q*exp(-2.77/Q*Ew); // Decrease it for true positives
    LOG(INFO) << "Output energy Ew(X1, X2) for label Y: " << Y << " = " << Ew << ", Loss = " << L;
    if(L > 1) loss += 1;
    else loss += L;

    sub += feature_size;
  }
  loss /= num;

  DLOG(INFO) << "Loss = " << loss;
  return loss;
}

template<typename Dtype>
void SiameseLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  
  // What to do here??
  LOG(INFO) << *(*bottom[0]);
  LOG(INFO) << *(top[0]);
  
}


INSTANTIATE_CLASS(SiameseLossLayer);


}  // namespace caffe
