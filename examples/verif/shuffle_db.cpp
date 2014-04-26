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
#include <algorithm>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>

#include "caffe/proto/caffe.pb.h"

using boost::shared_ptr;

#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

int main(int argc, char** argv) {
	if (argc != 3) {
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
	std::vector<std::string> data;
	char buf[32];
	int cnt = 0;
	while(iter->Valid()){
		data.push_back(iter->value().ToString());
		cnt ++;
		if(cnt % 10000 == 0)
			LOG(INFO) << "Read " << cnt;
		iter->Next();
	}
	LOG(INFO) << "records " << data.size();

	iter.reset();

	std::random_shuffle(data.begin(), data.end());

	options.create_if_missing = true;
	options.error_if_exists = true;
	status = leveldb::DB::Open(
			options, argv[2], &db_temp);
	CHECK(status.ok()) << "Failed to open leveldb "
		<< argv[2] << std::endl << status.ToString();
	db.reset(db_temp);
	leveldb::WriteOptions write_options;
	write_options.sync = false;
	for(int i=0;i<data.size();i++){
		if(i % 10000 == 0)
			LOG(INFO) << i;
		snprintf(buf, 32, "%08d", i);
		db->Put(leveldb::WriteOptions(), std::string(buf), data[i]);
	}

	LOG(INFO) << "DONE";

	return 0;
}

