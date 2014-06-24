// Copyright 2013 Yangqing Jia
// This program converts a set of images to a leveldb by storing them as Datum
// proto buffers.
// Usage:
//    convert_imageset ROOTFOLDER/ LISTFILE DB_NAME [0/1]
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....
// if the last argument is 1, a random shuffle will be carried out before we
// process the file lines.

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>

#include <fstream>  // NOLINT(readability/streams)
#include <sstream>
#include <iostream>
#include <iomanip>
#include <string>

#include <utility>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;
// NOLINT(build/namespaces)
using std::pair;
using std::string;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 7) {
    std::cout
        << "Convert a set of images to the leveldb format used as input for Caffe.\n\n"
        "Usage:\n\n"
        "./create_imageset_rgbd dataset_file  save_db_path save_database_information width height random_shuffle[0 or 1]\n\n"
        "datset_file                   file containing information about the dataset (label >> filepath >> depthpath >> tx >> ty >> tz >> q1 >> q2 >> q3 >> q4)\n"
        "save_db_name                  path to which the database will be written. Only contains labels and images\n"
        "save_database_information     complementary information written about the database (label >> filepath >> depthpath >> tx >> ty >> tz >> q1 >> q2 >> q3 >> q4)"
        "width, height\n"
        "random_shuffle\n";
    return 0;
  }

  std::cout << "Converting " << argv[1] << " to leveldb " << argv[2] << std::endl;

  std::istringstream ssw(argv[4]), ssh(argv[5]);
  int width, height;
  if (!(ssw >> width) || !(ssh >> height))
    LOG(ERROR)<< "Invalid number " << argv[4] << "x" << argv[5];
  LOG(INFO)<< "Images will be of size " << width <<"x" << height << ". They will be resized if needed (cv::resize)";

  LOG(INFO)<< "Reading from input file " << argv[1];
  std::ifstream infile(argv[1]);
  string filepath, depthpath;
  float tx, ty, tz, q1, q2, q3, q4;
  string dump;
  int label;

  std::string line;
  std::vector<std::string> lines;
  while (getline(infile, line)) {
    // Only keep lines that aren't comments
    if (line.substr(0, 1) != "#")
      lines.push_back(line);
  }

  if (argc == 7 && argv[6][0] == '1') {
    // randomly shuffle data
    LOG(INFO)<< "Shuffling data";
    std::random_shuffle(lines.begin(), lines.end());
  }
  LOG(INFO)<< "Retrieved information about " << lines.size() << " images.";
  LOG(INFO)<< "Reading image files and writing data to leveldb. Dataset contains pairs of label[image ID, read from input file] => Datum[image data]";

  leveldb::DB* db;
  leveldb::Options options;
  options.max_open_files = 10000;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  DLOG(INFO)<< "Opening leveldb " << argv[2];
  leveldb::Status status = leveldb::DB::Open(options, argv[2], &db);
  CHECK(status.ok()) << "Failed to open leveldb " << argv[2];

  std::ofstream extra_db;
  extra_db.open(argv[3]);

  string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  leveldb::WriteBatch* batch = new leveldb::WriteBatch();
  int data_size;
  bool data_size_initialized = false;
  for (auto line : lines) {
    std::string label_str;
    std::istringstream ss(line);
    // # id rgb depth tx ty tz q1 q2 q3 q4
    ss >> label >> filepath >> depthpath >> tx >> ty >> tz >> q1 >> q2 >> q3
        >> q4;

    std::ostringstream label_ss;
    label_ss << std::setfill('0') << std::setw(8) << label;
    label_str = label_ss.str();

    extra_db << label_str << " " << filepath << " " << depthpath << " " << tx
             << " " << ty << " " << tz << " " << q1 << " " << q2 << " " << q3
             << " " << q4 << std::endl;

    LOG(INFO)<< "Processing image with label: " << label_str << ", file: " << filepath;

    if (!ReadImageToDatum(filepath, label, width, height, &datum)) {
      LOG(ERROR)<< "Failed to read image " << filepath << " to Datum";
      continue;
    }

    if (!data_size_initialized) {
      data_size = datum.channels() * datum.height() * datum.width();
      data_size_initialized = true;
    } else {
      const string& data = datum.data();
      CHECK_EQ(data.size(), data_size)<< "Incorrect data field size "
      << data.size();
    }

    string value;
    // get the value
    datum.SerializeToString(&value);
    batch->Put(label_str, value);
    if (++count % 1000 == 0) {
      db->Write(leveldb::WriteOptions(), batch);
      LOG(INFO)<< "Processed " << count << " files.";
      delete batch;
      batch = new leveldb::WriteBatch();
    }
  }

  // write the last batch
  if (count % 1000 != 0) {
    LOG(INFO)<< "Writing final batch of: " << count%1000 << " elements";
   db->Write(leveldb::WriteOptions(), batch);
  }
  LOG(INFO)<< "Successfully processed " << count << " files. " << std::endl
  << "Dataset has been written to " << argv[2] << std::endl
  << "Extra information matching the dataset has been written to " << argv[3];

  delete batch;
  delete db;
  return 0;
}
