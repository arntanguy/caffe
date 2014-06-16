#include <glog/logging.h>
#include <leveldb/db.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

#include <algorithm>
#include <iterator>

#include <Eigen/Core>


/**
 * Computes the distance (euclidian) between two descriptors
 */
template<typename Dtype>
class Distance {
 public:
  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, 1> Descriptor;
  virtual Dtype operator()(const Descriptor& d1, const Descriptor& d2) {
    return (d1 - d2).norm();
  }
};


/**
 * Reads feature database, dataset information
 * Compute feature distances
 * Write it to file so that other tools can read it
 */
template<typename Dtype, typename DistanceFunctor>
class ProcessDescriptors {
  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, 1> Descriptor;
  typedef Eigen::Matrix<Dtype, 3, 1> Vec3;
  typedef Eigen::Matrix<Dtype, 4, 1> Quaternion;

  const std::string& dataset_path_;

  std::vector<int> ids_;
  std::vector<Descriptor> descriptors_;
  std::vector<std::string> rgb_dataset_;
  std::vector<Vec3> position_;
  std::vector<Quaternion> quaternion_;

  leveldb::DB* db_;
  leveldb::Iterator* it_;

  DistanceFunctor d_;

  void read_descriptors() {
    /**
     * Determine feature descriptor size
     */
    int feature_size = 0;
    it_->SeekToFirst();
    if (it_->Valid()) {
      std::vector<Dtype> v;
      std::istringstream iss(it_->value().ToString());

      // Iterate over the istream, using >> to grab floats
      // and push_back to store them in the vector
      std::copy(std::istream_iterator<Dtype>(iss),
                std::istream_iterator<Dtype>(), std::back_inserter(v));
      feature_size = v.size();
    }
    LOG(INFO)<< "Feature size is " << feature_size;

    /**
     * Read feature descriptor
     */
    for (it_->SeekToFirst(); it_->Valid(); it_->Next()) {
      const std::string& value = it_->value().ToString();
      std::istringstream iss(value);

      Descriptor ev(feature_size);
      Dtype f;
      int i = 0;
      while (iss >> f) {
        ev[i++] = f;
      }
      descriptors_.push_back(ev);
    }
    LOG(INFO)<< "Extracted " << descriptors_.size() << " descriptors.";
  }

  void read_dataset(const std::string& dataset_path)
  {
    std::ifstream ss(dataset_path);
    if (ss.is_open()) {
      // id rgb depth tx ty tz q1 q2 q3 q4
      int id;
      std::string rgb, depth;
      Dtype tx, ty, tz, q1, q2, q3, q4;
      // discard first line
      getline(ss, rgb);
      Vec3 translation;
      Quaternion quaternion;
      while(ss >> id >> rgb >> depth >> tx >> ty >> tz >> q1 >> q2 >> q3 >> q4) {
        ids_.push_back(id);
        rgb_dataset_.push_back(rgb);
        translation << tx, ty, tz;
        quaternion << q1, q2, q3, q4;
        position_.push_back(translation);
        quaternion_.push_back(quaternion);
      }
      LOG(INFO) << "Read " << position_.size() << " entries.";
    }

  }

  void compute_distances(const std::string& result_file) {
    std::ofstream rss(result_file);
    rss << "#DATASET " << dataset_path_ << "\n";
    rss << "#geometric_distance feature_distance id" << "\n";

    const Descriptor& ref_desc = descriptors_[0];
    const Vec3& ref_pos = position_[0];
    const Quaternion& ref_quaternion = quaternion_[0];

    int l = std::min(descriptors_.size(), ids_.size());
    for (int i = 0; i < l; ++i) {
      const Descriptor& d = descriptors_[i];
      const Vec3& pos = position_[i];
      const Quaternion& quat = quaternion_[i];

      Dtype feature_dist = d_(ref_desc, d);
      Dtype geometric_dist_t = (pos-ref_pos).norm();
      Dtype geometric_dist = geometric_dist_t * (1 - ref_quaternion.dot(quat) * ref_quaternion.dot(quat));
      LOG(INFO) << feature_dist << " " << geometric_dist;
      rss << geometric_dist << " " << feature_dist << " " << ids_[i] << std::endl;

    }
  }

public:
  ProcessDescriptors(const std::string& features_leveldb_path, const std::string& dataset_info, const std::string& result_file) : dataset_path_(dataset_info) {
    leveldb::Options options;
    options.error_if_exists = false;
    options.create_if_missing = false;
    options.max_open_files = 100;
    LOG(INFO) << "Using features database " << features_leveldb_path;
    leveldb::Status status = leveldb::DB::Open(options, features_leveldb_path, &db_);
    CHECK(status.ok()) << "Failed to open leveldb " << features_leveldb_path;

    it_ = db_->NewIterator(leveldb::ReadOptions());

    read_descriptors();
    read_dataset(dataset_info);
    compute_distances(result_file);
  }
  ~ProcessDescriptors() {
    delete it_;
    delete db_;
  }
};

int main(int argc, char **argv) {
  ::google::InitGoogleLogging(argv[0]);

  if (argc < 4) {
    std::cout
        << "Computes distance between features"
        << std::endl
        << "Usage feature_distance features_leveldb detailled_dataset_txt result_txt"
        << std::endl;
    return 1;
  }
  ProcessDescriptors<float, Distance<float>>(argv[1], argv[2], argv[3]);
  return 0;
}
