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
Dtype VerificationLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LOG(INFO) << "Forward_gpu";
  static int thr_update=0;
  if(thr_update != 100) {
    thr_update++;
  } else {
    thr_update=0;
 	  Dtype thr = this->CalcThreshold(true); 
    //this->accuracy_layer->SetThreshold(thr); 
    LOG(INFO) << "new_thr: " << thr; 
    this->ResetDistanceStat(); 
  }

  const Dtype* feat_1 = bottom[0]->gpu_data();
  const Dtype* feat_2 = bottom[1]->gpu_data();
  const Dtype* label = bottom[2]->cpu_data();

  int num = bottom[0]->num();

  Dtype* bottom_diff1 = diffy1_.mutable_gpu_data();
  Dtype* bottom_diff2 = diffy2_.mutable_gpu_data();

  int count = bottom[0]->count();
  CHECK_EQ(count, diffy1_.count());
  CHECK_EQ(count, diffy2_.count());
  //y1 - y2
  caffe_gpu_sub(count, feat_1, feat_2, bottom_diff1);
  caffe_gpu_sub(count, feat_2, feat_1, bottom_diff2);

  const int feat_len = bottom[0]->channels();
  Dtype loss(0.);

  bool die = false;
  for (int i = 0; i < bottom[0]->num(); ++i) {
    bool l = (static_cast<int>(label[i]) != 0);
    int offset = i*feat_len;
    Dtype norm2 = 0;
    caffe_gpu_dot(feat_len, bottom_diff1+offset,
        bottom_diff1+offset, &norm2);
    distance_.push_back(sqrt(norm2));
    same_.push_back(l);
    if(l) {
      /* nothing */
      //LOG(INFO) << i << " " << norm2;
      //LOG(INFO) << "1";
      loss += 0.5 * norm2;
    } else {
      //LOG(INFO) << i << " " << norm2;
      //LOG(INFO) << "0";
      Dtype dw = sqrt(norm2);
      if(dw > M_){
        CUDA_CHECK(cudaMemset(bottom_diff1+offset,0,
              sizeof(Dtype)*feat_len));
        CUDA_CHECK(cudaMemset(bottom_diff2+offset,0,
              sizeof(Dtype)*feat_len));
      } else {
        loss += 0.5 * (M_ - dw) * (M_ - dw);

        Dtype t = Dtype(1.0) - M_ / dw;
        caffe_gpu_scal(feat_len, t, bottom_diff1+offset);
        caffe_gpu_scal(feat_len, t, bottom_diff2+offset);
      }
    }
    die |= (norm2 > 1e6);	
  }

  return Dtype(loss/num);
}

template <typename Dtype>
  void VerificationLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, 
      const vector<bool>& propagate_down,
      vector<Blob<Dtype>*>* bottom) {
    const int num = (*bottom)[0]->num();
    const int count = (*bottom)[0]->count();
    Dtype* bottom_diff1 = diffy1_.mutable_gpu_data();
    Dtype* bottom_diff2 = diffy2_.mutable_gpu_data();
    //Add gradien to original
    if(propagate_down[0]) {
      Dtype* _bottom_diff1 = (*bottom)[0]->mutable_gpu_diff();
      Dtype* _bottom_diff2 = (*bottom)[2]->mutable_gpu_diff();
      caffe_gpu_axpy(count, LAMDA_ / num, bottom_diff1, _bottom_diff1);
      caffe_gpu_axpy(count, LAMDA_ / num, bottom_diff2, _bottom_diff2);
    }
  }

INSTANTIATE_CLASS(VerificationLossLayer);


}  // namespace caffe
