// Copyright 2014 Yuheng Chen

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void VerificationAccuracyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
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

  Dtype M2 = M*M;
  for (int i = 0; i < num; ++i) {
	int l1 = static_cast<int>(bottom_label1[i]);
	int l2 = static_cast<int>(bottom_label2[i]);
	int offset = i*dim;
	Dtype norm2(0.);
	caffe_gpu_dot(dim, diffy+offset, diffy+offset, &norm2);
	//LOG(INFO) << l1 << ' ' << l2 << ' ' << norm2 << ' ' << M2;
	if(l1 == l2 && norm2 <= M2)
		accuracy++;
	else if(l1 != l2 && norm2 > M2)
		accuracy++;
  }
  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
  (*top)[0]->mutable_cpu_data()[1] = logprob / num;

}

INSTANTIATE_CLASS(VerificationAccuracyLayer);
}
