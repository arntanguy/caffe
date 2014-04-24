// Copyright 2014 Yuheng Chen

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include <cstdio>
#include <cstdlib>

using std::string;

namespace caffe {

template <typename Dtype>
ShuffleDataLayer<Dtype>::~ShuffleDataLayer<Dtype>() {
}

template <typename Dtype>
void ShuffleDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype scale = this->layer_param_.scale();
  CHECK_EQ(bottom.size(), 0) << "Shuffle Data Layer takes no input blobs.";
  CHECK_EQ(top->size(), 2) << "Shuffle Data Layer takes two blobs as output.";
  leveldb::DB* db_temp;
  leveldb::Options options;
  options.create_if_missing = false;
  options.max_open_files = 100;
  LOG(INFO) << "Opening leveldb " << this->layer_param_.source();
  leveldb::Status status = leveldb::DB::Open(
      options, this->layer_param_.source(), &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb "
      << this->layer_param_.source() << std::endl << status.ToString();
  db_.reset(db_temp);
  iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
  iter_->SeekToFirst();
  
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(iter_->value().ToString());
(*top)[0]->Reshape(
        this->layer_param_.batchsize(), datum.channels(), datum.height(),
        datum.width());
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  (*top)[1]->Reshape(this->layer_param_.batchsize(), 1, 1, 1);
  // datum size
  datum_channels_ = datum.channels();
  datum_height_ = datum.height();
  datum_width_ = datum.width();
  datum_size_ = datum.channels() * datum.height() * datum.width();

  int n = 0;
  iter_->SeekToFirst();
  int nd = this->layer_param_.data_count();
  CHECK(nd != 0);
  DATA_COUNT_ = nd;

  LOG(INFO) << "data count " << nd;
  LOG(INFO) << "Starting reading all";
  prefetch_data_.reset(new Blob<Dtype>(nd, datum_channels_, datum_height_, datum_width_));
  prefetch_label_.reset(new Blob<Dtype>(nd, 1, 1, 1));
  if(!this->layer_param_.share_data()){
	  int i = 0;
	  while(i < nd && iter_->Valid()){
		datum.ParseFromString(iter_->value().ToString());
		Dtype *data_ptr = prefetch_data_->mutable_cpu_data() + datum_size_*i;
		prefetch_label_->mutable_cpu_data()[i] = datum.label();
        	for (int j = 0; j < datum_size_; ++j) {
			data_ptr[j] = datum.float_data(j) * scale;

		}
		i ++;
		iter_->Next();
	  }
	  CHECK_EQ(nd, i);
	  LOG(INFO) << "Read Done";
  }else{
	  LOG(INFO) << "skip read";
  }
  FILE *f = fopen(this->layer_param_.source_list().c_str(), "r");
  CHECK(f != NULL);
  int r = fscanf(f, "%d", &n);
  CHECK_EQ(r, 1);
  LOG(INFO) << n << " records in shuffle list";
  idx_.resize(n);
  for(int i=0;i<n;i++){
	  r = fscanf(f, "%d", &idx_[i]);
	  CHECK_EQ(r, 1);
	  CHECK(idx_[i] < nd);
  }
  fclose(f);

  current_ = 0;
}

template <typename Dtype>
void ShuffleDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
	CHECK(prefetch_data_.get() != NULL);
	CHECK(prefetch_label_.get() != NULL);
	const Dtype *ptr = prefetch_data_->cpu_data();
	for(int i=0;i<(*top)[0]->num();i++){
		int idx = idx_[current_];
		memcpy((*top)[0]->mutable_cpu_data() + i * datum_size_, ptr + idx * datum_size_, sizeof(Dtype)*datum_size_);
		(*top)[1]->mutable_cpu_data()[i] = prefetch_label_->cpu_data()[idx];
		current_++;
		if(current_ >= idx_.size())
			current_ = 0;
	}
}


template <typename Dtype>
Dtype ShuffleDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(ShuffleDataLayer);
}
