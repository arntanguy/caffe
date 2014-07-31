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
#include <lmdb.h>
#include <sys/stat.h>

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
  if (argc < 8) {
    std::cout
        << "Convert a set of images to the lmdb format used as input for Caffe.\n\n"
        "Usage:\n\n"
        "./create_imageset_rgbd dataset_file  save_db_path save_database_information width height random_shuffle[0 or 1]\n\n"
        "datset_file                   file containing information about the dataset (label >> filepath >> depthpath >> tx >> ty >> tz >> q1 >> q2 >> q3 >> q4)\n"
        "save_db_name                  path to which the database will be written. Only contains labels and images\n"
        "save_database_information     complementary information written about the database (label >> filepath >> depthpath >> tx >> ty >> tz >> q1 >> q2 >> q3 >> q4)"
        "width, height\n"
        "db_backend[leveldb or lmdb]    backend to use\n"
        "random_shuffle\n";
    return 0;
  }
  
  std::string input_file = argv[1];
  std::string save_db_name = argv[2];
  std::string extra_db_str = argv[3];
  std::string width_str = argv[4];
  std::string height_str = argv[5];
  std::string db_backend = argv[6];
  
  std::cout << "Converting " << input_file << " to " << db_backend << " " << save_db_name << std::endl;

  std::istringstream ssw(width_str), ssh(height_str);
  int width, height;
  if (!(ssw >> width) || !(ssh >> height))
    LOG(ERROR)<< "Invalid number " << width_str << "x" << height_str;
  LOG(INFO)<< "Images will be of size " << width <<"x" << height << ". They will be resized if needed (cv::resize)";

  LOG(INFO)<< "Reading from input file " << input_file;
  std::ifstream infile(input_file);
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

  if (argc == 8 && argv[7][0] == '1') {
    // randomly shuffle data
    LOG(INFO)<< "Shuffling data";
    std::random_shuffle(lines.begin(), lines.end());
  }
  LOG(INFO)<< "Retrieved information about " << lines.size() << " images.";
  LOG(INFO)<< "Reading image files and writing data to leveldb. Dataset contains pairs of label[image ID, read from input file] => Datum[image data]";

  // Open new db
  // lmdb
  MDB_env *mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;


  // leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.max_open_files = 10000;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  leveldb::WriteBatch* batch = NULL;

  if(db_backend == "leveldb") {
    DLOG(INFO)<< "Opening leveldb " << save_db_name;
    leveldb::Status status = leveldb::DB::Open(options, save_db_name, &db);
    CHECK(status.ok()) << "Failed to open leveldb " << save_db_name;
    batch = new leveldb::WriteBatch();
  } else if(db_backend == "lmdb") {
    LOG(INFO) << "Opening lmdb " << save_db_name;
    CHECK_EQ(mkdir(save_db_name.c_str(), 0744), 0)
        << "mkdir " << save_db_name << "failed";
    CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
        << "mdb_env_set_mapsize failed";
    CHECK_EQ(mdb_env_open(mdb_env, save_db_name.c_str(), 0, 0664), MDB_SUCCESS)
        << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
        << "mdb_open failed";
  } else {
    LOG(FATAL) << "Unknown db backend " << db_backend;
  }


  std::ofstream extra_db;
  extra_db.open(extra_db_str);

  std::string root_folder(input_file);
  Datum datum;
  int count = 0;
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

    //LOG(INFO)<< "Processing image with label: " << label_str << ", file: " << filepath;

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

    if ( db_backend == "leveldb" ) {
      batch->Put(label_str, value);
    } else if ( db_backend == "lmdb" ) {
      mdb_data.mv_size = value.size();
      mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
      mdb_key.mv_size = label_str.size();
      mdb_key.mv_data = reinterpret_cast<void*>(&label_str[0]);
      CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
          << "mdb_put failed";
    } else {
      LOG(FATAL) << "Unknown db backend " << db_backend;
    }

    if (++count % 1000 == 0) {
      if(db_backend == "leveldb") {
        db->Write(leveldb::WriteOptions(), batch);
        delete batch;
      } else if (db_backend == "lmdb") {  // lmdb
        CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
            << "mdb_txn_commit failed";
        CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
            << "mdb_txn_begin failed";
      } else {
        LOG(FATAL) << "Unknown db backend " << db_backend;
      }

      LOG(INFO)<< "Processed " << count << " files.";
      batch = new leveldb::WriteBatch();
    }
  }

  // write the last batch
  if (count % 1000 != 0) {
    LOG(INFO)<< "Writing final batch of: " << count%1000 << " elements";
    if ( db_backend == "leveldb" ) {
      db->Write(leveldb::WriteOptions(), batch);
      delete batch;
      delete db;
    } else if ( db_backend == "lmdb" ) {
      CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
      mdb_close(mdb_env, mdb_dbi);
      mdb_env_close(mdb_env);
    } else {
      LOG(FATAL) << "Unknown db backend " << db_backend;
    }
    LOG(INFO)<< "Successfully processed " << count << " files. " << std::endl
        << "Dataset has been written to " << save_db_name << std::endl
        << "Extra information matching the dataset has been written to " << argv[3];
  }

  return 0;
}
