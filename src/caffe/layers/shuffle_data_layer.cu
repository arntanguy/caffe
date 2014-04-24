// Copyright 2013 Yangqing Jia

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
void ShuffleDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
	//XXX opt it
  // Copy the data
/*
  CUDA_CHECK(cudaMemcpy((*top)[0]->mutable_gpu_data(),
      prefetch_data_->cpu_data(), sizeof(Dtype) * prefetch_data_->count(),
      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy((*top)[1]->mutable_gpu_data(),
      prefetch_label_->cpu_data(), sizeof(Dtype) * prefetch_label_->count(),
      cudaMemcpyHostToDevice));
      */
	CHECK(prefetch_data_.get() != NULL);
	CHECK(prefetch_label_.get() != NULL);
	//gather data in CPU
	const Dtype *ptr = prefetch_data_->cpu_data();
	for(int i=0;i<(*top)[0]->num();i++){
		int idx = idx_[current_];
		memcpy((*top)[0]->mutable_cpu_data() + i * datum_size_, ptr + idx * datum_size_, sizeof(Dtype)*datum_size_);
		(*top)[1]->mutable_cpu_data()[i] = prefetch_label_->cpu_data()[idx];
		current_++;
		if(current_ >= idx_.size())
			current_ = 0;
	}
	//sync
	(*top)[0]->gpu_data();
	(*top)[1]->gpu_data();
}

// The backward operations are dummy - they do not carry any computation.
template <typename Dtype>
Dtype ShuffleDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(ShuffleDataLayer);

}  // namespace caffe
