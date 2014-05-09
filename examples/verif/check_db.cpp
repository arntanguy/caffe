// Copyright Yuheng Chen 2013

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
#include <boost/shared_ptr.hpp>

#include "caffe/proto/caffe.pb.h"

using boost::shared_ptr;

#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

int main(int argc, char** argv) {
	if (argc != 4) {
		fprintf(stderr, "arg wrong\n");
		exit(1);
	} else {
		//google::InitGoogleLogging(argv[0]);
	}
	leveldb::DB* db_temp;
	leveldb::Options options;
	options.create_if_missing = false;
	options.max_open_files = 100;
	LOG(INFO) << "Opening leveldb " << argv[1];
	leveldb::Status status = leveldb::DB::Open(
			options, argv[1], &db_temp);
	CHECK(status.ok()) << "Failed to open leveldb "
		<< argv[1] << std::endl << status.ToString();

	shared_ptr<leveldb::DB> db(db_temp);
	shared_ptr<leveldb::Iterator> iter;

	iter.reset(db->NewIterator(leveldb::ReadOptions()));
	iter->SeekToFirst();
	caffe::Datum datum;
	int ph = atoi(argv[2]);
	int pw = atoi(argv[3]);
	CHECK(ph > 0 && pw > 0);
	int cnt = 0;
	float _min = 1e10, _max = -1e10;
	int minL = 1000000, maxL = -100000;
	while(iter->Valid()){
		datum.ParseFromString(iter->value().ToString());
		int l = datum.label();
		int s = datum.float_data_size();
		if(cnt % 10000 == 0)
			LOG(INFO) << "CHECK " << cnt;
		for(int i=0;i<s;i++){
			float f = datum.float_data(i);
			CHECK(f > -2.0f && f < 2.0f) << cnt;
			CHECK(f == f) << cnt;
			_min = std::min(_min, f);
			_max = std::max(_max, f);
		}
		minL = std::min(minL, l);
		maxL = std::max(maxL, l);

		cnt ++;
		iter->Next();
	}

	LOG(INFO) << "Samples: " << cnt;
	LOG(INFO) << "Label: " << minL << " to " << maxL;
	LOG(INFO) << "DONE min " << _min << ", max " << _max;

	return 0;
}

