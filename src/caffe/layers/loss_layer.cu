// Copyright 2014 Yuheng Chen

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template<typename Dtype>
void VerificationAccuracyLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  Dtype logprob = 0;
  const Dtype* bottom_data1 = bottom[0]->gpu_data();
  const Dtype* bottom_label1 = bottom[1]->cpu_data();
  const Dtype* bottom_data2 = bottom[2]->gpu_data();
  const Dtype* bottom_label2 = bottom[3]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();

  int count = bottom[0]->count();
  Dtype* diffy = diffy_.mutable_gpu_data();
  caffe_gpu_sub(count, bottom_data1, bottom_data2, diffy);

  Dtype M2 = M_ * M_;
  for (int i = 0; i < num; ++i) {
    int l1 = static_cast<int>(bottom_label1[i]);
    int l2 = static_cast<int>(bottom_label2[i]);
    int offset = i * dim;
    Dtype norm2(0.);
    caffe_gpu_dot(dim, diffy + offset, diffy + offset, &norm2);
    //LOG(INFO) << l1 << ' ' << l2 << ' ' << norm2 << ' ' << M2;
    if (l1 == l2 && norm2 <= M2)
      accuracy++;
    else if (l1 != l2 && norm2 > M2)
      accuracy++;
  }
  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
  (*top)[0]->mutable_cpu_data()[1] = logprob / num;

}

template<typename Dtype>
void VerificationAccuracyLayerSiamese<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Forward_cpu(bottom, top);
//
//  Dtype accuracy = 0;
//  Dtype logprob = 0;
//  const Dtype* bottom_data1 = bottom[0]->gpu_data();
////  const Dtype* bottom_data1 = bottom[0]->cpu_data();
//  const Dtype* bottom_label1 = bottom[1]->cpu_data();
//  const Dtype* bottom_data2 = bottom[2]->gpu_data();
////  const Dtype* bottom_data2 = bottom[2]->cpu_data();
//  const Dtype* bottom_label2 = bottom[3]->cpu_data();
//  int num = bottom[0]->num();
//  int dim = bottom[0]->count() / bottom[0]->num();
//  LOG(INFO) << "num: " << num << ", dim: " << dim;
////  for(int i=0; i<dim; i++) {
////      LOG(INFO) << "bottom: " << bottom[0]->cpu_data()[i] << ", " << bottom[2]->cpu_data()[i];
////    }
//
//
//  int count = bottom[0]->count();
//  Dtype* diffy = diffy_.mutable_gpu_data();
//  caffe_gpu_sub(count, bottom_data1, bottom_data2, diffy);
//
//  Dtype M2 = M_ * M_;
//  Dtype debug_cumulative_norm=Dtype(0.);
//  for (int i = 0; i < num; ++i) {
//    int l1 = static_cast<int>(bottom_label1[i]);
//    int l2 = static_cast<int>(bottom_label2[i]);
//    int offset = i * dim;
//    Dtype norm2(0.);
//    caffe_gpu_dot(dim, diffy + offset, diffy + offset, &norm2);
//    debug_cumulative_norm += norm2;
//    std::vector<std::pair<int, int> >::const_iterator it =
//        correspondance_labels_.begin();
//    bool same = false;
////  LOG(INFO) << "Looking for pair " << l1 << " " << l2;
//    while (it != correspondance_labels_.end()) {
//      if (it->first == l2 && it->second == l1
//          || it->first == l1 && it->second == l2) {
//        same = true;
//        break;
//      }
//      it++;
//    }
//
//    //LOG(INFO) << l1 << ' ' << l2 << ' ' << norm2 << ' ' << M2;
//    //if(same) LOG(INFO) << "Positive pair " << l1 <<", " << l2;
//    //else LOG(INFO) << "Negative pair " << l1 <<", " << l2;
//    if (same && norm2 <= M2) {
//      //LOG(INFO) << "norm2: " << norm2 << ", M^2: " << M2;
//      accuracy++;
//    }
//    else if(!same && norm2 > M2) {
//      //LOG(INFO) << "norm2: " << norm2 << ", M^2: " << M2;
//      accuracy++;
//    }
//  }
//  LOG(INFO) << "TOTAL CUMULATIVE NORM " << debug_cumulative_norm;
//      // LOG(INFO) << "Accuracy: " << accuracy;
//  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
//  (*top)[0]->mutable_cpu_data()[1] = logprob / num;

}

INSTANTIATE_CLASS(VerificationAccuracyLayer);
INSTANTIATE_CLASS(VerificationAccuracyLayerSiamese);

}
