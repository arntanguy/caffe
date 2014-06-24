// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template<typename Dtype>
Dtype VerificationLossLayer<Dtype>::CalcThreshold(bool update) {
  int i, j, is, id, is_ = 0, id_ = 0;
  Dtype th, th_c, s, d, f;
  int n = same_.size();
  CHECK_EQ(n, distance_.size());
  if (!n)
    return M_;
  for (auto same : same_) {
    if (same)
      is_++;
    else
      id_++;
  }

  Dtype stat[3];
  stat[0] = 1.0;
  stat[1] = 0.5;
  stat[2] = 0.5;
  th = -1.0;

  for (i = 0; i < 400; i++) {
    th_c = i * 0.1;
    is = 0;
    id = 0;
    for (j = 0; j < n; j++) {
      if (same_[j] && distance_[j] > th_c)
        is++;
      else if (!same_[j] && distance_[j] <= th_c)
        id++;
    }
    s = (Dtype) is / (2 * is_);
    d = (Dtype) id / (2 * id_);
    f = s + d;
    if (f < stat[0]) {
      stat[0] = f;
      stat[1] = s;
      stat[2] = d;
      th = th_c;
    }
  }
  LOG(INFO)<< "margin: " << th << " ("
  << stat[0] << ", " << stat[1]
  << ", " << stat[2] << ")";

  if (update)
    SetThreshold(th);
  return th;
}

template<typename Dtype>
void VerificationLossLayer<Dtype>::ReadCorrespondancesFile() {
  //XXX: should not be hardcoded
  correspondance_labels_.clear();
  const std::string &corr_file =
      "/media/DATA/Datasets/SLAM_LOOP/loop_closures_positive.txt";
  LOG(INFO)<< "Reading correspondance labels for closed loops from file " << corr_file;
  int l1, l2;
  int read = 0;
  std::string line;
  std::vector<std::pair<float, float> > ssss;

  std::ifstream infile(corr_file);
  CHECK(infile.is_open()) << "Impossible to open file " << corr_file;
  std::getline(infile, line);
  while (infile >> l1 >> l2) {
    //LOG(INFO)<< "pair: " << l1 << " " << l2;
    correspondance_labels_.push_back(std::make_pair(l1, l2));
    read++;
  }
  infile.close();
  LOG(INFO)<< "Read "<< correspondance_labels_.size() << " positive correspondances";
}

template<typename Dtype>
void VerificationLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                         vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 4)<< "VerificationLoss Layer takes four blobs as input.";
  CHECK_EQ(top->size(), 0) << "VerificationLoss Layer takes no blob as output.";

  diffy1_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diffy2_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  M_ = this->layer_param_.dual_threshold();
  LAMDA_ = this->layer_param_.dual_lamda();

  ResetDistanceStat();
  LOG(INFO) << "Initial: threshold " << M_ << ", " << "lamda: " << LAMDA_;
}

template<typename Dtype>
void VerificationLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
}

template<typename Dtype>
Dtype VerificationLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* feat_1 = (*bottom)[0]->cpu_data();
  const Dtype* feat_2 = (*bottom)[2]->cpu_data();
  const Dtype* label_1 = (*bottom)[1]->cpu_data();
  const Dtype* label_2 = (*bottom)[3]->cpu_data();

  //Dtype *diffy_ptr = diffy_.mutable_cpu_data();

  Dtype* bottom_diff1 = diffy1_.mutable_cpu_data();
  Dtype* bottom_diff2 = diffy2_.mutable_cpu_data();

  int num = (*bottom)[0]->num();
  int count = (*bottom)[0]->count();
  //y1 - y2
  caffe_sub(count, feat_1, feat_2, bottom_diff1);
  caffe_sub(count, feat_2, feat_1, bottom_diff2);

  const int feat_len = (*bottom)[0]->channels();

  for (int i = 0; i < (*bottom)[0]->num(); ++i) {
    int l1 = static_cast<int>(label_1[i]);
    int l2 = static_cast<int>(label_2[i]);
    LOG(INFO)<< l1 << "\t" << l2;
    int offset = i * feat_len;

    auto it =
        std::find_if(
            correspondance_labels_.begin(),
            correspondance_labels_.end(),
            [&l1, &l2](const std::pair<int, int>& p)
            { return ((p.first == l1 && p.second == l2) || (p.first == l2 && p.second == l1));});
    // If (l1,l2) is a loop closure do nothing
    if (it != correspondance_labels_.end()) {
      LOG(INFO)<< "(" << l1 << ", " << l2 << ") is a loop-closure";
      /* nothing */
    } else {
      LOG(INFO) << "(" << l1 << ", " << l2 << ") is not a loop-closure";
      /* Update */
      Dtype norm2 = caffe_cpu_dot(feat_len, bottom_diff1+offset, bottom_diff1+offset);
      Dtype norm = sqrt(norm2);
      if(norm > M_) {
        memset(bottom_diff1+offset,0, sizeof(Dtype)*feat_len);
        memset(bottom_diff2+offset,0, sizeof(Dtype)*feat_len);
      } else {
        norm = (M_ - norm) / (norm+Dtype(FLT_MIN));
        caffe_scal(feat_len, -norm, bottom_diff1+offset);
        caffe_scal(feat_len, -norm, bottom_diff2+offset);
      }
    }
  }

  //Add gradien to original
  Dtype* _bottom_diff1 = (*bottom)[0]->mutable_cpu_diff();
  Dtype* _bottom_diff2 = (*bottom)[2]->mutable_cpu_diff();
#if 0
  for(int i=0;i<(*bottom)[0]->count();i++) {
    printf("%d %f %f\n", num, _bottom_diff1[i], bottom_diff1[i] / num);
  }
#endif

  // Scale down gradient
  caffe_axpy(count, LAMDA_ / num, bottom_diff1, _bottom_diff1);
  caffe_axpy(count, LAMDA_ / num, bottom_diff2, _bottom_diff2);
  return Dtype(0.);
}

INSTANTIATE_CLASS(VerificationLossLayer);

}  // namespace caffe
