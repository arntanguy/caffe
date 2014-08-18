// Copyright 2014 BVLC and contributors.

#include <stdio.h>  // for snprintf
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include "lmdb.h"
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;
// NOLINT(build/namespaces)

int count_lmdb_entries(const std::string& db_name);
template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int count_lmdb_entries(const std::string& db_name)
{
  // LMDB
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;

  CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
  CHECK_EQ(mdb_env_open(mdb_env_, db_name.c_str(), MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
      << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
      << "mdb_open failed";
  CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
      << "mdb_cursor_open failed";
  LOG(INFO) << "Opening lmdb " << db_name; 
  CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
           MDB_SUCCESS) << "mdb_cursor_get failed";
  int count = 1;
  while(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT) == 0) {
    ++count;
  }

  mdb_cursor_close(mdb_cursor_);
  mdb_close(mdb_env_, mdb_dbi_);
  mdb_txn_abort(mdb_txn_);
  mdb_env_close(mdb_env_);
  return count;
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  const int num_required_args = 6;
  if (argc < num_required_args) {
    /**
     * Num mini batches : number of image batch to process
     * batch size defined in prototxt
     */
    std::cout <<    "Extracts feature using a trained CNN\n\n"
    "Usage: demo_extract_features  trained_net trained_net_proto blob_name result_leveldb  num_batches  [CPU/GPU]  [DEVICE_ID=0]\n\n"
    "trained_net          trained network binary file\n"
    "trained_net_proto    trained network prototxt description file\n"
    "source_leveldb       leveldb dataset to extract from\n"
    "blob_name            name of the blob to extract features from\n"
    "result_leveldb       path to the leveldb where the results will be saved\n";
    return 1;
  }
  const std::string pretrained_binary_proto(argv[1]);
  const std::string feature_extraction_proto(argv[2]);
  const std::string source_leveldb = argv[3];
  const std::string extract_feature_blob_name = argv[4];
  const std::string save_feature_leveldb_name = argv[5];

  int arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
    LOG(ERROR)<< "Using GPU";
    uint device_id = 0;
    if (argc > arg_pos + 1) {
      device_id = atoi(argv[arg_pos + 1]);
      CHECK_GE(device_id, 0);
    }
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }
  Caffe::set_phase(Caffe::TEST);



  // Expected prototxt contains at least one data layer such as
  //  the layer data_layer_name and one feature blob such as the
  //  fc7 top blob to extract features.
  /*
   layers {
   name: "data_layer_name"
   type: DATA
   data_param {
   source: "/path/to/your/images/to/extract/feature/images_leveldb"
   mean_file: "/path/to/your/image_mean.binaryproto"
   batch_size: 128
   crop_size: 227
   mirror: false
   }
   top: "data_blob_name"
   top: "label_blob_name"
   }
   layers {
   name: "drop7"
   type: DROPOUT
   dropout_param {
   dropout_ratio: 0.5
   }
   bottom: "fc7"
   top: "fc7"
   }
   */

  NetParameter net_param;
  ReadProtoFromTextFile(feature_extraction_proto, &net_param);
  if(net_param.layers_size() > 0) {
    const int DATA_LAYER_IDX=0;
    LOG(INFO) << "Layers exists";
    LOG(INFO) << "source: " << net_param.mutable_layers(DATA_LAYER_IDX)->mutable_layer()->source();
    net_param.mutable_layers(DATA_LAYER_IDX)->mutable_layer()->set_source(source_leveldb.c_str());
  }

  shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(net_param));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  CHECK(feature_extraction_net->has_blob(extract_feature_blob_name))
      << "Unknown feature blob name " << extract_feature_blob_name
      << " in the network " << feature_extraction_proto;

  //const shared_ptr<DataLayer<Dtype> >& data_layer =  boost::static_pointer_cast<DataLayer<Dtype>>(feature_extraction_net->layer_by_name("verif_dual_001"));
  const shared_ptr< Layer<Dtype> >& data_layer =  feature_extraction_net->layer_by_name("data");
  // FIXME: get number of element in dataset!
  //int data_count = 100; //data_layer->layer_param().data_count();
  int data_count = count_lmdb_entries(source_leveldb);
  LOG(INFO) << "Data count: " << data_count;
  int batch_size = data_layer->layer_param().data_param().batch_size();
  int num_mini_batches = ceil((float)data_count/(float)batch_size);
  LOG(INFO) << "Data size: " <<  data_count << ", batch size: " << batch_size << ", number of batches: "<< num_mini_batches;

  arg_pos++;
  LOG(INFO)<< "Output leveldb " << save_feature_leveldb_name;
  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  LOG(INFO)<< "Opening leveldb " << save_feature_leveldb_name;
  leveldb::Status status = leveldb::DB::Open(options,
                                             save_feature_leveldb_name.c_str(),
                                             &db);
  CHECK(status.ok()) << "Failed to open leveldb " << save_feature_leveldb_name;


  LOG(ERROR)<< "Extacting Features";

  leveldb::WriteBatch* batch = new leveldb::WriteBatch();
  vector<Blob<Dtype>*> input_vec;
  int image_index = 0;
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    LOG(INFO) << "Batch " << batch_index+1 << "/" << num_mini_batches;
    feature_extraction_net->Forward(input_vec);
    const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
        ->blob_by_name(extract_feature_blob_name);
    // Extract batch_size features, or less for the last batch
    const int num_features = (data_count-image_index > batch_size) ? batch_size : data_count-image_index; 
    // Size of a feature
    const int dim_features = feature_blob->count() / batch_size;
    Dtype* feature_blob_data;
    DLOG(INFO)<< "dim features: " << dim_features;

    // Extract each feature in the batch
    for (int n = 0; n < num_features; ++n) {
      feature_blob_data = feature_blob->mutable_cpu_data()
          + feature_blob->offset(n);
      std::ostringstream data_stream;
      data_stream.precision(std::numeric_limits<double>::digits10);
      // Extract feature from batch
      for (int d = 0; d < dim_features - 1; ++d) {
        data_stream << feature_blob_data[d] << " ";
      }
      data_stream << feature_blob_data[dim_features - 1];
      //LOG(INFO)<< "DATA( " << batch_index << ", " << n <<  "): "<< data_stream.str();
      std::ostringstream key_str_stream;
      key_str_stream << std::setfill('0') << std::setw(8) << image_index;
      batch->Put(string(key_str_stream.str()), data_stream.str());
      ++image_index;
      if (image_index % 1000 == 0) {
        db->Write(leveldb::WriteOptions(), batch);
        LOG(ERROR)<< "Extracted features of " << image_index <<
        " query images.";
        delete batch;
        batch = new leveldb::WriteBatch();
      }
    }  // for (int n = 0; n < num_features; ++n)
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
  // write the last batch
  if (image_index % 1000 != 0) {
    db->Write(leveldb::WriteOptions(), batch);
    LOG(INFO)<< "Extracted features of " << image_index <<
    " query images.";
  }

  delete batch;
  delete db;
  LOG(INFO)<< "Successfully extracted the features!";
  return 0;
}




int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

