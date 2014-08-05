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
  // Initial energy is null
  Ew = 0;
  // XXX: should be set to the upper bound of Ew
  // Upper bound: (max_feature_descriptor_value-min_feature_descriptor_value) * feature_size
  // Here assumes max=1, min=0
  // ||x+y|| <= ||x||+||y||
  Q = 2* bottom[0]->channels(); 
  dd = new Dtype[bottom[0]->channels()*bottom[0]->num()];
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
   * F(X1): output descriptor for input X1
   * F(X2): output descriptor for input X2
   * TODO
   * Ew(X1, X2) = || F(X1) - F(X2) ||1   : output energy of the network
   * 
   **/

  DLOG(INFO) << "Number of input blobs: " << bottom.size();
  const int num = bottom[0]->num();
  const int feature_size = bottom[0]->channels();
  DLOG(INFO) << "Ouput blob info: " << *bottom[0];

  Dtype* F_x1 = bottom[0]->mutable_cpu_data();
  Dtype* F_x2 = bottom[1]->mutable_cpu_data();
  // label
  const Dtype* labels = bottom[2]->cpu_data();

  /**
   * Difference between the two descriptors:
   * d = W*(F(X1)-F(X2))
   **/
  Dtype *sub = diff_.mutable_cpu_data();

  Dtype *f1 = F_x1;
  Dtype *f2 = F_x2;

  static Dtype max_energy = 0;
  static Dtype max_desc = 0;
  static Dtype min_desc = 0;
  static Dtype max_norm = 0;

  //LOG(INFO) << "Max energy (theory): " << Q;
  //LOG(INFO) << "Max energy (current): " << max_energy;
  //LOG(INFO) << "Min desc value (current): " << min_desc;
  //LOG(INFO) << "Max desc value (current): " << max_desc;

  Dtype loss = 0;
  for(int i=0; i< num; i++) {
    const int Y = labels[i];
    
    Dtype ddot = caffe_cpu_dot(feature_size, f1, f1);
    Dtype nnorm = sqrt(ddot); 
    caffe_cpu_scale(feature_size, Dtype(1)/nnorm, f1, f1);
    ddot = caffe_cpu_dot(feature_size, f2, f2);
    nnorm = sqrt(ddot); 
    caffe_cpu_scale(feature_size, Dtype(1)/nnorm, f2, f2);
    max_norm = std::max(nnorm, max_norm);

    caffe_sub(feature_size, f1, f2, sub);

    for (int desc=i*feature_size; desc < (i+1)*feature_size; desc++) {
      max_desc=std::max(F_x1[desc], max_desc);
      max_desc=std::max(F_x2[desc], max_desc);
      min_desc=std::min(F_x1[desc], min_desc);
      min_desc=std::min(F_x2[desc], min_desc);
    }
    // Norm L1 (sum of absolute values)
    // Dtype Ew = caffe_cpu_asum(feature_size, sub);
    
    // Norm L2 (sqrt of dot product absolute values)
    Dtype d_dot = caffe_cpu_dot(feature_size, sub, sub);
    Ew = sqrt(d_dot);
    max_energy = max(max_energy, Ew);

    Dtype L=0;
    if(Y == 1) {
      L  = Dtype(2)/Q * Ew*Ew;
    } else {
      L = Dtype(2)/Q*exp(Dtype(-2.27)/Q*Ew);
    }
    //LOG(INFO) << "Output energy (L2) Ew(X1, X2) for label " << Y << " = " << Ew << "/" << max_energy <<", Loss = " << L;
    //LOG(INFO) << "max norm: " << nnorm;
    loss += L;
    sub += feature_size;
    f1 += feature_size;
    f2 += feature_size;
  }
  loss /= num;
  return loss;
}

template<typename Dtype>
void SiameseLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  
  // What to do here??
  CHECK_EQ(bottom->size(), 3) << "Bottom size must be 3 blobs for backwards propagation of the siamese loss";
  DLOG(INFO) << "Number of input blobs: " << bottom->size();
  Blob<Dtype> *f1 = (*bottom)[0];
  Blob<Dtype> *f2 = (*bottom)[1];
  Blob<Dtype> *l = (*bottom)[2];
  const int num = f1->num();
  const int count = f1->count();
  const int feature_size = f1->channels();
  DLOG(INFO) << "Ouput blob info: " << *f1;

  Dtype *grad1 = f1->mutable_cpu_diff();
  Dtype *grad2 = f2->mutable_cpu_diff();

  // label
  const Dtype* labels = l->cpu_data();

  Dtype *g1 = grad1;
  Dtype *g2 = grad2;

  Dtype *d_mult = dd;
  const Dtype *d = diff_.cpu_data();
  caffe_mul(count, d, d, dd);

  for(int i=0; i< num; i++) {
    const int Y = labels[i];
    if(Y == 1) { // genuine
      caffe_cpu_scale(feature_size, Dtype(4)/Q, d_mult, g1);
      caffe_cpu_scale(feature_size, Dtype(4)/Q, d_mult, g2);
    } else { // impostor
      Dtype d_dot = caffe_cpu_dot(feature_size, d, d);
      const Dtype Ew = sqrt(d_dot);
      const Dtype Ci = Dtype(-4.27)/Ew*exp(Dtype(-2.27/Q)*Ew);
      caffe_cpu_scale(feature_size, Ci, d, g1);
      caffe_cpu_scale(feature_size, Ci, d, g2);
    } 
    d_mult += feature_size;
    d += feature_size;
  }
     //caffe_cpu_axpby(
     //    count,              // count
     //    Dtype(1) / num,         // alpha
     //    grad1,                   // a
     //    Dtype(0),                           // beta
     //    grad1);  // b
     //caffe_cpu_axpby(
     //    count,              // count
     //    Dtype(1) / num,         // alpha
     //    grad2,                   // a
     //    Dtype(0),                           // beta
     //    grad2);  // b

}


INSTANTIATE_CLASS(SiameseLossLayer);


}  // namespace caffe
