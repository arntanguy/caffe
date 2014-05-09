// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <limits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/vision_layers.hpp"

using std::max;

namespace caffe {


template <typename Dtype>
__global__ void DropoutForward(const int n, const Dtype* in,
    const unsigned int* mask, const int channels, const int mask_size, 
    const unsigned int threshold, const Dtype *scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    int num = index / channels / mask_size;
    mask += num * mask_size;
    int i = index % mask_size;
    out[index] = in[index] * (mask[i] < threshold) * scale[num];
  }
}

template <typename Dtype>
void DropoutGroupLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int mask_size = bottom[0]->width() * bottom[0]->height();
  if (Caffe::phase() == Caffe::TRAIN) {
    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
    const Dtype *scale_ptr = reinterpret_cast<const Dtype*>
	    (scale_->gpu_data());
    DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, (unsigned int*)rand_vec_->gpu_data(),
	bottom[0]->channels(), mask_size,
	uint_thres_, scale_ptr, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    CUDA_CHECK(cudaMemcpy(top_data, bottom_data,
        count * sizeof(Dtype), cudaMemcpyDeviceToDevice));
  }
}

template <typename Dtype>
__global__ void DropoutBackward(const int n, const Dtype* in_diff,
    const unsigned int* mask, const int channels, const int mask_size, 
    const unsigned int threshold, const Dtype *scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    int num = index / channels / mask_size;
    mask += num * mask_size;
    int i = index % mask_size;
    out_diff[index] = in_diff[index] * scale[num] * (mask[i] < threshold);
  }
}

template <typename Dtype>
Dtype DropoutGroupLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  CHECK(Caffe::phase() == Caffe::TRAIN);
  if (propagate_down) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const unsigned int* mask = (unsigned int*)rand_vec_->gpu_data();
    const int count = (*bottom)[0]->count();
    const int mask_size = top[0]->width() * top[0]->height();
    // NOLINT_NEXT_LINE(whitespace/operators)

    const Dtype *scale_ptr = reinterpret_cast<const Dtype*>
	    (scale_->gpu_data());
    DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top[0]->channels(), mask_size, uint_thres_,
	scale_ptr, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
  return Dtype(0);
}

INSTANTIATE_CLASS(DropoutGroupLayer);

}  // namespace caffe
