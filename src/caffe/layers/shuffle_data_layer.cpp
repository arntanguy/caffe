// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>

#include <string>
#include <vector>

#include <fstream>
#include <sstream>
#include <iomanip>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"

using std::string;

namespace caffe {

// This function is used to create a pthread that prefetches the data.
template <typename Dtype>
void ShuffleDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(prefetch_data_);
  Dtype* top_data = prefetch_data_->mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (output_labels_) {
    top_label = prefetch_label_->mutable_cpu_data();
  }
  const Dtype scale = this->layer_param_.data_param().scale();
  const int crop_size = this->layer_param_.data_param().crop_size();
  const bool mirror = this->layer_param_.data_param().mirror();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
        << "set at the same time.";
  }
  // datum scales
  const int channels = datum_channels_;
  const int height = datum_height_;
  const int width = datum_width_;
  const int size = datum_size_;
  const Dtype* mean = data_mean_.cpu_data();
  for (int item_id = 0; item_id < batch_size_; ++item_id) {
    int current_id = current_id_+item_id;
    if(current_id >= idx_.size()) {
        LOG(INFO) << "Restarting data prefetching from start.";
        current_id_ = 0;
        current_id = 0;
    }
    //
    // XXX: maybe store the labels as int in the dataset.
    // However, if done that way it would depend on the encoding used for
    // int (endian...).
    const int id = idx_[current_id];
    std::ostringstream label_ss;
    label_ss << std::setfill('0') << std::setw(8) << id;
    std::string key = label_ss.str();
    //LOG(INFO) << "Id: " << current_id << ", Image " << key << " from channel " << channel_;

    // get a blob
    switch (this->layer_param_.data_param().backend()) {
      case DataParameter_DB_LEVELDB:
        {
          std::string value;
          leveldb::Status s = db_->Get(leveldb::ReadOptions(), key, &value);
          CHECK(s.ok()) << "Failed to read image with id " << key << ": " << s.ToString();
          datum.ParseFromString(value);
          break;
        }
      case DataParameter_DB_LMDB:
        {
          MDB_val mdb_key;
          mdb_key.mv_size = key.size();
          mdb_key.mv_data = reinterpret_cast<void*>(&key[0]);
          CHECK_EQ(mdb_get(mdb_txn_, mdb_dbi_, &mdb_key, &mdb_value_), MDB_SUCCESS);
          datum.ParseFromArray(mdb_value_.mv_data,
                               mdb_value_.mv_size);
          break;
        }
      default:
        LOG(FATAL) << "Unknown database backend";
    }

    const string& data = datum.data();
    if (crop_size) {
      CHECK(data.size()) << "Image cropping only support uint8 data";
      int h_off, w_off;
      // We only do random crop when we do training.
      if (phase_ == Caffe::TRAIN) {
        h_off = PrefetchRand() % (height - crop_size);
        w_off = PrefetchRand() % (width - crop_size);
      } else {
        h_off = (height - crop_size) / 2;
        w_off = (width - crop_size) / 2;
      }
      if (mirror && PrefetchRand() % 2) {
        // Copy mirrored version
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < crop_size; ++h) {
            for (int w = 0; w < crop_size; ++w) {
              int top_index = ((item_id * channels + c) * crop_size + h)
                              * crop_size + (crop_size - 1 - w);
              int data_index = (c * height + h + h_off) * width + w + w_off;
              Dtype datum_element =
                  static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
              top_data[top_index] = (datum_element - mean[data_index]) * scale;
            }
          }
        }
      } else {
        // Normal copy
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < crop_size; ++h) {
            for (int w = 0; w < crop_size; ++w) {
              int top_index = ((item_id * channels + c) * crop_size + h)
                              * crop_size + w;
              int data_index = (c * height + h + h_off) * width + w + w_off;
              Dtype datum_element =
                  static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
              top_data[top_index] = (datum_element - mean[data_index]) * scale;
            }
          }
        }
      }
    } else {
      // we will prefer to use data() first, and then try float_data()
      if (data.size()) {
        for (int j = 0; j < size; ++j) {
          Dtype datum_element =
              static_cast<Dtype>(static_cast<uint8_t>(data[j]));
          top_data[item_id * size + j] = (datum_element - mean[j]) * scale;
        }
      } else {
        for (int j = 0; j < size; ++j) {
          top_data[item_id * size + j] =
              (datum.float_data(j) - mean[j]) * scale;
        }
      }
    }

    if (output_labels_) {
      // LOG(INFO) << "Label: " << lc_[current_id];
      top_label[item_id] = lc_[current_id];
      //if(lc_[current_id] == 0) {
      //  top_label[item_id] = -1;
      //} else {
      //  top_label[item_id] = 1;
      //}
    }
  }
}

template <typename Dtype>
ShuffleDataLayer<Dtype>::~ShuffleDataLayer<Dtype>() {
  JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    break;  // do nothing
  case DataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

template<typename Dtype>
void ShuffleDataLayer<Dtype>::ReadShuffleList()
{
  LOG(INFO) << "Reading Shuffle List from: "<< this->layer_param_.data_param().shuffle_param().source_list();
  std::ifstream ss(this->layer_param_.data_param().shuffle_param().source_list().c_str());
  CHECK(ss.is_open()) << "Failed to open shuffle list!";
  int npairs = 0;
  ss >> npairs;
  LOG(INFO) << npairs;
  idx_.reserve(npairs);
  lc_.reserve(npairs);
  int id[2];
  int lc;
  int ind=0;
  while(ss >> id[0] >> id[1] >> lc) {
    // Check if indexes are valid
    //CHECK_GE(max_index_in_dataset, id1);
    //CHECK_GE(max_index_in_dataset, id2);
    //if(id1 < max_index_in_dataset && id2 < max_index_in_dataset) {
      // Add them to pair list
      idx_.push_back(id[channel_]);
      lc_.push_back(lc);
    //}
    ++ind;
  }
  LOG(INFO) << idx_.size() << " ids added to shuffle list.";
}

template <typename Dtype>
void ShuffleDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  if (top->size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  batch_size_ = this->layer_param_.data_param().batch_size();
  LOG(INFO) << "Set batch size: " << batch_size_;

  // start with channel 0
  channel_ = this->layer_param_.data_param().shuffle_param().channel();
  CHECK(channel_ == 0 || channel_ == 1) << "shuffle_param.channel must be 0 or 1!";
  current_id_ = 0;

  ReadShuffleList();

  // Initialize DB
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    {
    leveldb::DB* db_temp;
    leveldb::Options options;
    options.create_if_missing = false;
    options.max_open_files = 100;
    LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
    leveldb::Status status = leveldb::DB::Open(
        options, this->layer_param_.data_param().source(), &db_temp);
    CHECK(status.ok()) << "Failed to open leveldb "
                       << this->layer_param_.data_param().source() << std::endl
                       << status.ToString();
    db_.reset(db_temp);
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    iter_->SeekToFirst();
    }
    break;
  case DataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    LOG(INFO) << "rand_skip not implemented";
    //unsigned int skip = caffe_rng_rand() %
    //                    this->layer_param_.data_param().rand_skip();
    //LOG(INFO) << "Skipping first " << skip << " data points.";
    //while (skip-- > 0) {
    //  switch (this->layer_param_.data_param().backend()) {
    //  case DataParameter_DB_LEVELDB:
    //    iter_->Next();
    //    if (!iter_->Valid()) {
    //      iter_->SeekToFirst();
    //    }
    //    break;
    //  case DataParameter_DB_LMDB:
    //    LOG(INFO) << "LMDB backend not implemented";
    //    break;
    //  default:
    //    LOG(FATAL) << "Unknown database backend";
    //  }
    //}
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    datum.ParseFromString(iter_->value().ToString());
    break;
  case DataParameter_DB_LMDB:
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // image
  int crop_size = this->layer_param_.data_param().crop_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(batch_size_,
                       datum.channels(), crop_size, crop_size);
    prefetch_data_.reset(new Blob<Dtype>(
        batch_size_, datum.channels(),
        crop_size, crop_size));
  } else {
    (*top)[0]->Reshape(
        batch_size_, datum.channels(),
        datum.height(), datum.width());
    prefetch_data_.reset(new Blob<Dtype>(
        batch_size_, datum.channels(),
        datum.height(), datum.width()));
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  if (output_labels_) {
    (*top)[1]->Reshape(batch_size_, 1, 1, 1);
    prefetch_label_.reset(
        new Blob<Dtype>(batch_size_, 1, 1, 1));
  }
  // datum size
  datum_channels_ = datum.channels();
  datum_height_ = datum.height();
  datum_width_ = datum.width();
  datum_size_ = datum.channels() * datum.height() * datum.width();
  CHECK_GT(datum_height_, crop_size);
  CHECK_GT(datum_width_, crop_size);
  // check if we want to have mean
  if (this->layer_param_.data_param().has_mean_file()) {
    const string& mean_file = this->layer_param_.data_param().mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.channels(), datum_channels_);
    CHECK_EQ(data_mean_.height(), datum_height_);
    CHECK_EQ(data_mean_.width(), datum_width_);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  if (output_labels_) {
    prefetch_label_->mutable_cpu_data();
  }
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void ShuffleDataLayer<Dtype>::CreatePrefetchThread() {
  //LOG(INFO) << "Prefetch from channel " << channel_;
  phase_ = Caffe::phase();
  const bool prefetch_needs_rand = (phase_ == Caffe::TRAIN) &&
      (this->layer_param_.data_param().mirror() ||
       this->layer_param_.data_param().crop_size());
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }
  // Create the thread.
  CHECK(!StartInternalThread()) << "Pthread execution failed";
}

template <typename Dtype>
void ShuffleDataLayer<Dtype>::JoinPrefetchThread()
{
  CHECK(!WaitForInternalThreadToExit()) << "Pthread joining failed!";
}

template <typename Dtype>
unsigned int ShuffleDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
void ShuffleDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_->count(), prefetch_data_->cpu_data(),
             (*top)[0]->mutable_cpu_data());

  if (output_labels_) {
    caffe_copy(prefetch_label_->count(), prefetch_label_->cpu_data(),
               (*top)[1]->mutable_cpu_data());
  }

  current_id_ += batch_size_;

  // Start a new prefetch thread
  CreatePrefetchThread();
}

INSTANTIATE_CLASS(ShuffleDataLayer);

}  // namespace caffe
