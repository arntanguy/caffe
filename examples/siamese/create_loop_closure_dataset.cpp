#include <glog/logging.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

#include <algorithm>
#include <iterator>

#include <Eigen/Core>
#include <cmath>

template<typename Dtype>
class CreateLoopClosures {

  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, 1> Descriptor;
  typedef Eigen::Matrix<Dtype, 3, 1> Vec3;
  typedef Eigen::Matrix<Dtype, 4, 1> Quaternion;

  std::vector<int> ids_;
  std::vector<Descriptor> descriptors_;
  std::vector<std::string> rgb_filepath_;
  std::vector<std::string> depth_filepath_;
  std::vector<Vec3> position_;
  std::vector<Quaternion> quaternion_;

  float rotation_;
  float translation_;
  int keyframe_step_;
  int keyframe_distance_;
  std::ifstream dataset_istream_;
  std::ofstream lc_positive_ofstream_;
  std::ofstream lc_negative_ofstream_;

  void read_dataset() {
    LOG(INFO)<< "Read dataset";
    // id rgb depth tx ty tz q1 q2 q3 q4
    int id;
    std::string rgb, depth;
    Dtype tx, ty, tz, q1, q2, q3, q4;
    // discard first line
    getline(dataset_istream_, rgb);
    Vec3 translation;
    Quaternion quaternion;
    while (dataset_istream_ >> id >> rgb >> depth >> tx >> ty >> tz >> q1 >> q2
        >> q3 >> q4) {
      ids_.push_back(id);
      rgb_filepath_.push_back(rgb);
      depth_filepath_.push_back(depth);
      translation << tx, ty, tz;
      quaternion << q1, q2, q3, q4;
      position_.push_back(translation);
      quaternion_.push_back(quaternion);
    }
    LOG(INFO) << "Read " << position_.size() << " entries.";
  }

  void find_loop_closures() {
    LOG(INFO) << "find_loop_closures";
    int nb_pos, nb_neg = 0;
    for(int i=0; i<ids_.size(); i++) {
      const Vec3& current_translation = position_[i];
      const Quaternion& current_rotation = quaternion_[i];

      std::vector<std::pair<int, int>> pairs;
      pairs.reserve(ids_.size()*ids_.size());
      for(int j=0; j<ids_.size(); j++) {
        // Skip frames that are too close to each other
        if(abs(j-i) >= keyframe_distance_) {
          const Vec3& other_translation = position_[j];
          const Quaternion& other_rotation = quaternion_[j];
          Dtype geom_distance_t = (current_translation - other_translation).norm();
          Dtype rotation_angle = std::abs(std::acos(2*std::pow(current_rotation.dot(other_rotation), 2)-1))*180.f/M_PI;
          //LOG(INFO) << "i, j, distance, angle: " << i <<", " <<j<<", " << geom_distance_t << ", " << rotation_angle;

          std::pair<int, int> pair(i, j);
          auto it = std::find_if(pairs.begin(),
                                 pairs.end(),
                                [&pair](const std::pair<int, int>& p)
                                { return (p.first == pair.first && p.second == pair.second) || (p.first == pair.second && p.second == pair.first); });
          if(it == pairs.end()) {
            pairs.push_back(pair);

            if(geom_distance_t < translation_ && rotation_angle < rotation_) {
              lc_positive_ofstream_ << i << " " << j << "\n";
              ++nb_pos;
            } else {
              lc_negative_ofstream_ << i << " " << j << "\n";
              ++nb_neg;
            }
          }
        }
      }

    }
    std::cout << "Done. " << nb_pos << " positive, " << nb_neg << " negatives" << std::endl;
  }

public:
  CreateLoopClosures() {
    rotation_ = 5;
    translation_ = .5;
    keyframe_step_ = 5;
    keyframe_distance_ = 5;

    dataset_istream_.open("dataset.txt");
    CHECK(dataset_istream_.is_open()) << "Could not open input dataset";
    lc_positive_ofstream_.open("loop_closures_positive.txt");
    CHECK(lc_positive_ofstream_.is_open()) << "Could not open loop_closure_positive";
    lc_negative_ofstream_.open("loop_closures_negative.txt");
    CHECK(lc_negative_ofstream_.is_open()) << "Could not open loop_closure_nagative";

    read_dataset();
    find_loop_closures();
  }

  ~CreateLoopClosures() {
  }

};

int main(int argc, char **argv) {
  ::google::InitGoogleLogging(argv[0]);
  CreateLoopClosures<float> cl;
}
