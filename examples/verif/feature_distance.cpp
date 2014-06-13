#include <glog/logging.h>
#include <leveldb/db.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <limits>

#include <algorithm>
#include <iterator>

#include <Eigen/Core>

#include <string>

// Multiply two doubles
template<typename Dtype>
class Distance {
 public:
  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, 1> Descriptor;
  Distance() {
  }
  ;
  virtual Dtype operator()(const Descriptor& d1, const Descriptor& d2) {
    return (d1 - d2).norm();
  }
};

template<typename Dtype, typename DistanceFunctor>
int process_descriptors(int argc, char **argv) {
  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, 1> Descriptor;

  const std::string& features_db_path = argv[1];
  LOG(INFO) << "Using features database " << features_db_path;

  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = false;
  options.create_if_missing = false;
  options.max_open_files = 100;
  leveldb::Status status = leveldb::DB::Open(options, features_db_path, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << features_db_path;

  leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());

  /**
   * Determine feature descriptor size
   */
  int feature_size = 0;
  it->SeekToFirst();
  if (it->Valid()) {
    std::vector<Dtype> v;
    std::istringstream iss(it->value().ToString());

    // Iterate over the istream, using >> to grab floats
    // and push_back to store them in the vector
    std::copy(std::istream_iterator<Dtype>(iss), std::istream_iterator<Dtype>(),
              std::back_inserter(v));
    feature_size = v.size();
  }
  LOG(INFO) << "Feature size is " << feature_size;

  /**
   * Read feature descriptor
   */
  std::vector<Descriptor> descriptors;
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    const std::string& value = it->value().ToString();
    std::istringstream iss(value);

    Descriptor ev(feature_size);
    Dtype f;
    int i = 0;
    while (iss >> f) {
      ev[i++] = f;
    }
    descriptors.push_back(ev);
  }
  LOG(INFO) << "Extracted " << descriptors.size() << " descriptors.";

  Descriptor ref = descriptors[0];
  DistanceFunctor d;
  for (Descriptor v : descriptors) {
    LOG(INFO) << d(ref, v);
  }
  return 0;
}

int main(int argc, char **argv) {
  ::google::InitGoogleLogging(argv[0]);

  if (argc < 2) {
    std::cout << "Computes distance between features" << std::endl
              << "Usage feature_distance features_leveldb" << std::endl;
    return 1;
  }
  return process_descriptors<float, Distance<float> >(argc, argv);
}
