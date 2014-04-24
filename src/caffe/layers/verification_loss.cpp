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
void VerificationLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 4) << "VerificationLoss Layer takes four blobs as input.";
  CHECK_EQ(top->size(), 0) << "VerificationLoss Layer takes no blob as output.";

  diffy1_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diffy2_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  M = Dtype(0.);
  ALPHA = Dtype(0.);
}

template <typename Dtype>
void VerificationLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
}

template <typename Dtype>
Dtype VerificationLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* feat_1 = (*bottom)[0]->cpu_data();
  const Dtype* feat_2 = (*bottom)[2]->cpu_data();
  const Dtype* label_1 = (*bottom)[1]->cpu_data();
  const Dtype* label_2 = (*bottom)[3]->cpu_data();
  
  //Dtype *diffy_ptr = diffy_.mutable_cpu_data();

  Dtype* bottom_diff1 = diffy1_.mutable_cpu_data();
  Dtype* bottom_diff2 = diffy2_.mutable_cpu_data();

  int count = (*bottom)[0]->count();
  //y1 - y2
  caffe_sub(count, feat_1, feat_2, bottom_diff1);
  caffe_sub(count, feat_2, feat_1, bottom_diff2);

  const int feat_len = (*bottom)[0]->channels();

  for (int i = 0; i < (*bottom)[0]->num(); ++i) {
	int l1 = static_cast<int>(label_1[i]);
	int l2 = static_cast<int>(label_2[i]);
	int offset = i*feat_len;
	if(l1 == l2){
		/* nothing */
	}else{
		Dtype norm2 = caffe_cpu_dot(feat_len, bottom_diff1+offset, bottom_diff1+offset);
		Dtype norm = sqrt(norm2);
		if(norm > M){
			memset(bottom_diff1+offset,0, sizeof(Dtype)*feat_len);
			memset(bottom_diff2+offset,0, sizeof(Dtype)*feat_len);
		}else{
			norm = (M - norm) / (norm+Dtype(FLT_MIN));
			caffe_scal(feat_len, -norm, bottom_diff1+offset);
			caffe_scal(feat_len, -norm, bottom_diff2+offset);
		}
	}
  }

  //Add gradien to original
  Dtype* _bottom_diff1 = (*bottom)[0]->mutable_cpu_diff();
  Dtype* _bottom_diff2 = (*bottom)[2]->mutable_cpu_diff();
  caffe_axpy(count, ALPHA, bottom_diff1, _bottom_diff1);
  caffe_axpy(count, ALPHA, bottom_diff2, _bottom_diff2);
  return Dtype(0.);
}


INSTANTIATE_CLASS(VerificationLossLayer);


}  // namespace caffe
