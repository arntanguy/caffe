// Copyright 2014 BVLC and contributors.

#include <stdio.h>  // for snprintf
#include <cuda_runtime.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
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

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
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
  const shared_ptr< Layer<Dtype> >& data_layer =  feature_extraction_net->layer_by_name("verif_dual_001");
  int data_count = data_layer->layer_param().data_count();
  int batch_size = data_layer->layer_param().batchsize();
  int num_mini_batches = ceil((float)data_count/(float)batch_size);
  LOG(INFO) << "Data size: " <<  data_count << ", batch size: " << batch_size << ", number of batches: "<< num_mini_batches;

  arg_pos++;
  LOG(INFO)<< "Input leveldb " << save_feature_leveldb_name;
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
  int num_bytes_of_binary_code = sizeof(Dtype);
  vector<Blob<Dtype>*> input_vec;
  int image_index = 0;
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    LOG(INFO) << "Batch " << batch_index+1 << "/" << num_mini_batches;
    feature_extraction_net->Forward(input_vec);
    const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
        ->blob_by_name(extract_feature_blob_name);
    int num_features = (data_count/batch_size > 0) ? batch_size : batch_size - data_count % batch_size + 1; //feature_blob->num();
    int dim_features = feature_blob->count() / num_features;
    Dtype* feature_blob_data;
    DLOG(INFO)<< "dim features: " << dim_features;
    for (int n = 0; n < num_features; ++n) {
      feature_blob_data = feature_blob->mutable_cpu_data()
          + feature_blob->offset(n);
      std::ostringstream data_stream;
      data_stream.precision(std::numeric_limits<double>::digits10);
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
    LOG(ERROR)<< "Extracted features of " << image_index <<
    " query images.";
  }

  delete batch;
  delete db;
  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}

