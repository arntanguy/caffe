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
void VerificationLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
}

template <typename Dtype>
Dtype VerificationLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	//TODO
  const Dtype* feat_1 = (*bottom)[0]->gpu_data();
  const Dtype* feat_2 = (*bottom)[2]->gpu_data();
  const Dtype* label_1 = (*bottom)[1]->cpu_data();
  const Dtype* label_2 = (*bottom)[3]->cpu_data();
  
  //Dtype *diffy_ptr = diffy_.mutable_gpu_data();

  Dtype* bottom_diff1 = diffy1_.mutable_gpu_data();
  Dtype* bottom_diff2 = diffy2_.mutable_gpu_data();

  int count = (*bottom)[0]->count();
  CHECK_EQ(count, diffy1_.count());
  CHECK_EQ(count, diffy2_.count());
  //y1 - y2
  caffe_gpu_sub(count, feat_1, feat_2, bottom_diff1);
  caffe_gpu_sub(count, feat_2, feat_1, bottom_diff2);

  const int feat_len = (*bottom)[0]->channels();
#if 0
  Dtype tmp;
  caffe_gpu_dot(feat_len, bottom_diff1+0,
		bottom_diff1+0, &tmp);
  LOG(INFO) << tmp;
#endif
  Dtype loss(0.);

  for (int i = 0; i < (*bottom)[0]->num(); ++i) {
	int l1 = static_cast<int>(label_1[i]);
	int l2 = static_cast<int>(label_2[i]);
	int offset = i*feat_len;
	if(l1 == l2){
		/* nothing */
		Dtype norm2 = 0;
		caffe_gpu_dot(feat_len, bottom_diff1+offset,
				bottom_diff1+offset, &norm2);
		//LOG(INFO) << i << " " << norm2;
		loss += sqrt(norm2);
	}else{
		Dtype norm2 = 0;
		caffe_gpu_dot(feat_len, bottom_diff1+offset,
				bottom_diff1+offset, &norm2);
		//LOG(INFO) << i << " " << norm2;
		Dtype norm = sqrt(norm2);
		if(norm > M_){
			CUDA_CHECK(cudaMemset(bottom_diff1+offset,0,
						sizeof(Dtype)*feat_len));
			CUDA_CHECK(cudaMemset(bottom_diff2+offset,0,
					sizeof(Dtype)*feat_len));
		}else{
			norm = (M_ - norm) / (norm+Dtype(FLT_MIN));
			norm = std::min(norm, Dtype(100.0));
			caffe_gpu_scal(feat_len, -norm, bottom_diff1+offset);
			caffe_gpu_scal(feat_len, -norm, bottom_diff2+offset);
			loss += norm*sqrt(norm2);
		}
	}
  }

  int num = (*bottom)[0]->num();
  //Add gradien to original
  Dtype* _bottom_diff1 = (*bottom)[0]->mutable_gpu_diff();
  Dtype* _bottom_diff2 = (*bottom)[2]->mutable_gpu_diff();
  caffe_gpu_axpy(count, LAMDA_ / num, bottom_diff1, _bottom_diff1);
  caffe_gpu_axpy(count, LAMDA_ / num, bottom_diff2, _bottom_diff2);

  return Dtype(LAMDA_ / num * loss); 
}

INSTANTIATE_CLASS(VerificationLossLayer);


}  // namespace caffe
