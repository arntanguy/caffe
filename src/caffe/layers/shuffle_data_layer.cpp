// Copyright 2014 Yuheng Chen

#include <leveldb/db.h>

#include <fstream>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/util/debug.hpp"

namespace caffe {

template<typename Dtype>
ShuffleDataLayer<Dtype>::~ShuffleDataLayer<Dtype>() {
}

template<typename Dtype>
void ShuffleDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                    vector<Blob<Dtype>*>* top) {
  LOG(INFO)<< "SetUp ShuffleDataLayer";
  const Dtype scale = this->layer_param_.scale();
  const Dtype bias = this->layer_param_.bias();
  CHECK_EQ(bottom.size(), 0)<< "Shuffle Data Layer takes no input blobs.";
  CHECK_EQ(top->size(), 2)<< "Shuffle Data Layer takes two blobs as output.";
  OUTPUT_CHANNEL_ = 0;
  leveldb::DB* db_temp;
  leveldb::Options options;
  options.create_if_missing = false;
  options.max_open_files = 100;
  LOG(INFO)<< "Opening leveldb " << this->layer_param_.source();
  leveldb::Status status = leveldb::DB::Open(options,
      this->layer_param_.source(),
      &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb " << this->layer_param_.source()
  << std::endl << status.ToString();
  db_.reset(db_temp);
  iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
  iter_->SeekToFirst();

  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(iter_->value().ToString());
  (*top)[0]->Reshape(this->layer_param_.batchsize(), datum.channels(),
      datum.height(), datum.width());
  LOG(INFO)<< "output data size: " << (*top)[0]->num() << ","
  << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
  << (*top)[0]->width();
  // label
  (*top)[1]->Reshape(this->layer_param_.batchsize(), 1, 1, 1);
  // datum size
  datum_channels_ = datum.channels();
  datum_height_ = datum.height();
  datum_width_ = datum.width();
  datum_size_ = datum.channels() * datum.height() * datum.width();

  LOG(INFO) << "Datum info:\n"
  << "\tChannels: " << datum_channels_ << "\n"
  << "\tWidth x Height: " << datum_width_ << "x" << datum_height_ << "\n"
  << "\tSize: " << datum_size_ << "\n";

  int n = 0;
  iter_->SeekToFirst();
  size_t nd = this->layer_param_.data_count();
  CHECK(nd != 0);
  DATA_COUNT_ = nd;

  LOG(INFO)<< "data count " << nd;
  LOG(INFO)<< "Starting reading all";
  //prefetch_data_.reset(new Blob<Dtype>(nd, datum_channels_, datum_height_, datum_width_));
  prefetch_data_.reset(new vector<Dtype>((size_t) nd * datum_size_));
  prefetch_label_.reset(new Blob<Dtype>(nd, 1, 1, 1));

  if (!this->layer_param_.share_data()) {
    size_t i = 0;
    Dtype *pd_hdr = &((*prefetch_data_)[0]);
    /**
     * Read one batch of nd
     *  elements
     */
    while (i < nd && iter_->Valid()) {
      datum.ParseFromString(iter_->value().ToString());
      Dtype *data_ptr = pd_hdr + datum_size_ * i;
      prefetch_label_->mutable_cpu_data()[i] = datum.label();

      for (int j = 0; j < datum_size_; ++j) {
        data_ptr[j] = (datum.data()[j] + bias) * scale;
      }

      DLOG(INFO)<< "Label: " << datum.label() << ", Data[0,1..n-1,n]: " << data_ptr[0] << ", "
      << data_ptr[1] << " ... " << data_ptr[datum_size_/sizeof(Dtype)-1] << ", " << data_ptr[datum_size_/sizeof(Dtype)];
      i++;
      iter_->Next();
    }
    /**
     * Read pairs of loop-closures (50%) and non loop-closures (50%)
     * from file.
     * Fill in the id_[] array with these pairs. The training algorithm will
     * select the appropriate one.
     */
    /*
     CHECK_EQ(nd, i);
     LOG(INFO) << "Read Done";
     FILE *f = fopen(this->layer_param_.source_list().c_str(), "r");
     CHECK(f != NULL);
     int r = fscanf(f, "%d", &n);
     CHECK_EQ(r, 1);
     LOG(INFO) << n << " records in shuffle list";
     idx_[0].reset(new vector<int>(n));
     idx_[1].reset(new vector<int>(n));
     for(int i=0;i<n;i++){
     int d1, d2;
     // Read pair from file
     r = fscanf(f, "%d%d", &d1, &d2);
     CHECK_EQ(r, 2);
     // Read ids in pair can't be greater than the number of images in the dataset!
     CHECK_GT(nd, d1);
     CHECK_GT(nd, d2);
     // Assign pair to the class
     (*idx_[0])[i] = d1;
     (*idx_[1])[i] = d2;
     }
     fclose(f);
     LOG(INFO) << "read list done"; */

    LOG(INFO) << "Reading Shuffle List: "<< this->layer_param_.source_list();
    std::ifstream ss(this->layer_param_.source_list().c_str());
    CHECK(ss.is_open()) << "Failed to open shuffle list!";
    int npairs = 0;
    ss >> npairs;
    idx_[0].reset(new vector<int>(npairs));
    idx_[1].reset(new vector<int>(npairs));
    int id1, id2;
    int ind=0;
    while(ss >> id1 >> id2) {
      // Check if indexes are valid
      CHECK_GE(nd, id1);
      CHECK_GE(nd, id2);
      if(id1 < nd && id2 < nd) {
      // Add them to pair list
      (*idx_[0])[ind] = id1;
      (*idx_[1])[ind] = id2;
      }
      ++ind;
    }
    LOG(INFO) << idx_[0]->size() << " pairs added to shuffle list." << (*idx_[0])[1];
  } else {
    LOG(INFO)<< "skip read";
  }
  iter_.reset();
  db_.reset();
  current_[0] = current_[1] = 0;
}

template<typename Dtype>
void ShuffleDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          vector<Blob<Dtype>*>* top) {
  DLOG(INFO)<< "ShuffleDataLayer::Forward_cpu with bottom size " << bottom.size() << " and top size " << top->size();
  CHECK(prefetch_data_.get() != NULL);
  CHECK(prefetch_label_.get() != NULL);
  CHECK(idx_[0].get() != NULL);
  const Dtype *ptr = &((*prefetch_data_)[0]);

  /**
   * Copy all images for the current batch
   */
  DLOG(INFO) << "Copying batch of " << (*top)[0]->num() << " images for output channel " << OUTPUT_CHANNEL_;
  for(size_t i=0;i<(*top)[0]->num();i++) {
    /**
     * OUTPUT_CHANNEL_ is set by the training algorithm.
     * It is 0 for the network, and 1 for the shadow network.
     * This allows to process data in batches while conserving the
     * matching pairs of images.
     */
    size_t idx = (*idx_[OUTPUT_CHANNEL_])[current_[OUTPUT_CHANNEL_]];
    //LOG(INFO) << "Copying batch image " << i+1 << " of " << (*top)[0]->num() << "-- output channel: " << OUTPUT_CHANNEL_ << ", pair number: " << current_[OUTPUT_CHANNEL_] << ", image id: " << idx;
    // dst, src, size
    memcpy((*top)[0]->mutable_cpu_data() + i * datum_size_, ptr + idx * datum_size_, sizeof(Dtype)*datum_size_);
    //(*top)[1]->mutable_cpu_data()[i] = prefetch_label_->cpu_data()[idx];

    // Label is current image id
    (*top)[1]->mutable_cpu_data()[i] = idx;

//#ifdef NDEBUG_GUI
//      float *img = new float[datum_size_];
//      memcpy(img, ptr + idx * datum_size_, sizeof(Dtype)*datum_size_);
//      displayImageFromData(img, datum_height_, datum_width_);
//      delete[] img;
//#endif

      current_[OUTPUT_CHANNEL_]++;
      if(current_[OUTPUT_CHANNEL_] >= idx_[OUTPUT_CHANNEL_]->size()) {
        current_[OUTPUT_CHANNEL_] = 0;
        LOG(INFO) << "Channel " << OUTPUT_CHANNEL_ << " rewind";
      }
    }
  }

template<typename Dtype>
Dtype ShuffleDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                            const bool propagate_down,
                                            vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(ShuffleDataLayer);
}
