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
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>

using boost::shared_ptr;

#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"
void convertColor(const cv::Mat& src, cv::Mat &dst){
	double min;
	double max;
	cv::minMaxIdx(src, &min, &max);
	LOG(INFO) << "MINMAX " << min << "," << max;
	cv::Mat adjMap;
	// expand your range to 0..255. Similar to histEq();
	src.convertTo(dst,CV_8UC1, 255 / (max-min), -min); 
//cv::applyColorMap(dst, dst, cv::COLORMAP_AUTUMN);
}

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
	while(iter->Valid()){
		datum.ParseFromString(iter->value().ToString());
		int l = datum.label();
		int size = datum.float_data_size();
		CHECK(size % (ph*pw) == 0);
		int ch = size / ph / pw;
		LOG(INFO) << "Lable " << l;
		cv::Mat out;
		if(ch == 3){
			cv::Mat mat[3];
			for(int i=0;i<3;i++){
				mat[i] = cv::Mat::zeros(ph, pw, CV_32FC1);
				float *p = (float*)mat[i].ptr();
				int base = ph * pw * i;
				for(int j=0;j<ph*pw;j++)
					p[j] = datum.float_data(base + j);
				convertColor(mat[i], mat[i]);
			}
			cv::merge(mat, 3, out);
		}else if(ch == 1){
			cv::Mat mat = cv::Mat::zeros(ph, pw, CV_32FC1);
			NOT_IMPLEMENTED;
		}else{
			LOG(ERROR) << "channel " << ch << " unsupported";
		}
		cv::imshow("img", out);
		cv::waitKey(0);

		iter->Next();
	}

	LOG(INFO) << "DONE";

	return 0;
}

