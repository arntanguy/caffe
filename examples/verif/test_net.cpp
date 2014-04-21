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
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
	if (argc < 3) {
		LOG(ERROR) << "argc";
		return -1;
	}

	cudaSetDevice(0);
	Caffe::set_phase(Caffe::TEST);

	Caffe::set_mode(Caffe::GPU);

	NetParameter test_net_param;
	ReadProtoFromTextFile(argv[1], &test_net_param);
	Net<float> caffe_test_net(test_net_param);
	NetParameter trained_net_param;
	caffe_test_net.ToProto(&trained_net_param, false);
	WriteProtoToBinaryFile(trained_net_param, argv[2]);


	return 0;
}
