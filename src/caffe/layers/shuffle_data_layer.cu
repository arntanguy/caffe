// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>
#include <sstream>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/util/debug.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
Dtype ShuffleDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  WaitForInternalThreadToExit();
  // Copy the data
  caffe_copy(prefetch_data_->count(), prefetch_data_->cpu_data(),
      (*top)[0]->mutable_gpu_data());

#if (DEBUG_GUI)
  const int batchsize = (*top)[0]->num();
  //const int channels = (*top)[0]->channels();
  std::ostringstream ss;
  ss << "ShuffleDataLayer, channel " << channel_;
  displayImageFromData(ss.str().c_str(), (*top)[0]->mutable_cpu_data(), datum_height_, datum_width_, batchsize, 3);
#endif

  if (output_labels_) {
    caffe_copy(prefetch_label_->count(), prefetch_label_->cpu_data(),
        (*top)[1]->mutable_gpu_data());
  }

  current_id_ += batch_size_; 

  // Start a new prefetch thread
  CreatePrefetchThread();
  return Dtype(0.);
}

INSTANTIATE_CLASS(ShuffleDataLayer);

}  // namespace caffe
