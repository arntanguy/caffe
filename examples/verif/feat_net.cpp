// Copyright 2013 Yangqing Jia
//
// This is a simple script that allows one to quickly test a network whose
// structure is specified by text format protocol buffers, and whose parameter
// are loaded from a pre-trained network.
// Usage:
//    test_net net_proto pretrained_net_proto iterations [CPU/GPU]

#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>

#include "caffe/caffe.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

template <typename Dtype>
static void save_blob(const string& fn, Blob<Dtype> *b){
	LOG(INFO) << "Saving " << fn;
	FILE *f = fopen(fn.c_str(), "wb");
	CHECK(f != NULL);
	fwrite(b->cpu_data(), sizeof(Dtype), b->count(), f);
	fclose(f);
}

int main(int argc, char** argv) {
  if (argc < 5) {
    LOG(ERROR) << "test_net net_proto pretrained_net_proto iterations output_dir"
        << " [CPU/GPU]";
    return 0;
  }

  Caffe::set_phase(Caffe::TEST);

  if (argc == 6 && strcmp(argv[5], "GPU") == 0) {
    LOG(ERROR) << "Using GPU";
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  NetParameter test_net_param;
  ReadProtoFromTextFile(argv[1], &test_net_param);
  Net<float> caffe_test_net(test_net_param);
  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(argv[2], &trained_net_param);
  caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

#if 0
  SolverState state;
  std::string state_file = std::string(argv[2]) + ".solverstate";
  ReadProtoFromBinaryFile(state_file, &state);
#endif

  int total_iter = atoi(argv[3]);
  LOG(ERROR) << "Running " << total_iter << "Iterations.";

  double test_accuracy = 0;
  vector<Blob<float>*> dummy_blob_input_vec;

  //save layer
  char output_dir[1024];
  int feature_layer_idx = -1;
  for(int i=0;i<caffe_test_net.layer_names().size();i++)
	  if(caffe_test_net.layer_names()[i] == "relu5"){
		  feature_layer_idx = i;
		  break;
	  }
  CHECK_NE(feature_layer_idx, -1);
  LOG(INFO) << "Feature layer: " << feature_layer_idx;

  Blob<float>* output = caffe_test_net.top_vecs()[feature_layer_idx][0];
  LOG(INFO) << "OUTPUT BLOB dim: " << output->num() << ' '
	  << output->channels() << ' '
	  << output->width() << ' '
	  << output->height();
  for (int i = 0; i < total_iter; ++i) {
    const vector<Blob<float>*>& result =
        caffe_test_net.Forward(dummy_blob_input_vec);

    sprintf(output_dir, "%s/feat_%05d", argv[4], i);
    save_blob(output_dir, output);

    //test_accuracy += result[0]->cpu_data()[0];
    //LOG(ERROR) << "Batch " << i << ", accuracy: " << result[0]->cpu_data()[0];
  }
  //test_accuracy /= total_iter;
  //LOG(ERROR) << "Test accuracy:" << test_accuracy;

  return 0;
}
