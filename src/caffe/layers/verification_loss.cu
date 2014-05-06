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

  bool die = false;
  for (int i = 0; i < (*bottom)[0]->num(); ++i) {
	int l1 = static_cast<int>(label_1[i]);
	int l2 = static_cast<int>(label_2[i]);
	int offset = i*feat_len;
	Dtype norm2 = 0;
	caffe_gpu_dot(feat_len, bottom_diff1+offset,
		bottom_diff1+offset, &norm2);
	distance_.push_back(sqrt(norm2));
	same_.push_back((l1 == l2) ? 1: 0);
	if(l1 == l2){
		/* nothing */
		//LOG(INFO) << i << " " << norm2;
		//LOG(INFO) << "1";
		loss += 0.5 * norm2;
	}else{
		//LOG(INFO) << i << " " << norm2;
		//LOG(INFO) << "0";
		Dtype dw = sqrt(norm2);
		if(dw > M_){
			CUDA_CHECK(cudaMemset(bottom_diff1+offset,0,
						sizeof(Dtype)*feat_len));
			CUDA_CHECK(cudaMemset(bottom_diff2+offset,0,
					sizeof(Dtype)*feat_len));
		}else{
			loss += 0.5 * (M_ - dw) * (M_ - dw);

			Dtype t = Dtype(1.0) - M_ / dw;
			caffe_gpu_scal(feat_len, t, bottom_diff1+offset);
			caffe_gpu_scal(feat_len, t, bottom_diff2+offset);
		}
	}
	die |= (norm2 > 1e6);	
  }
#if 0
  char buf[256];
  static int dumpi = 0;
  sprintf(buf, "dump_%d.bin", dumpi);
  dumpi+=1;
  FILE *f = fopen(buf, "wb");
#if 1
  fwrite(diffy1_.cpu_data(), diffy1_.count(),
		  sizeof(Dtype), f  );
  fwrite(diffy2_.cpu_data(), diffy2_.count(),
		  sizeof(Dtype), f  );
#else
  fwrite((*bottom)[0]->cpu_diff(), (*bottom)[0]->count(),
		  sizeof(Dtype), f  );
  fwrite((*bottom)[2]->cpu_diff(), (*bottom)[0]->count(),
		  sizeof(Dtype), f  );
#endif
  fclose(f);
  LOG(INFO) << "DUMP " << dumpi;
#endif

  int num = (*bottom)[0]->num();
  //Add gradien to original
  Dtype* _bottom_diff1 = (*bottom)[0]->mutable_gpu_diff();
  Dtype* _bottom_diff2 = (*bottom)[2]->mutable_gpu_diff();
  caffe_gpu_axpy(count, LAMDA_ / num, bottom_diff1, _bottom_diff1);
  caffe_gpu_axpy(count, LAMDA_ / num, bottom_diff2, _bottom_diff2);

  return Dtype(loss / num); 
}

INSTANTIATE_CLASS(VerificationLossLayer);


}  // namespace caffe
