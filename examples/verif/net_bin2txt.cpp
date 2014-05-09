#include <google/protobuf/text_format.h>
#include <glog/logging.h>
#include <leveldb/db.h>
#include <cassert>

#include <stdint.h>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;
int main(int argc, char *argv[])
{
	CHECK_EQ(argc, 3);
	NetParameter net;
	ReadProtoFromBinaryFile(argv[1], &net);
	WriteProtoToTextFile(net, argv[2]);
	return 0;
}
