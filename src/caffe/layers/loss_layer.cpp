// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>
#include <fstream>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

const float kLOG_THRESHOLD = 1e-20;

template<typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::SetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2)<< "Loss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "Loss Layer takes no output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
  << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
}

template<typename Dtype>
Dtype MultinomialLogisticLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  const Dtype* bottom_label = (*bottom)[1]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  int num = (*bottom)[0]->num();
  int dim = (*bottom)[0]->count() / (*bottom)[0]->num();
  memset(bottom_diff, 0, sizeof(Dtype) * (*bottom)[0]->count());
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    Dtype prob = max(bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
    loss -= log(prob);
    bottom_diff[i * dim + label] = -1. / prob / num;
  }
  return loss / num;
}

// TODO: implement the GPU version for multinomial loss

template<typename Dtype>
void InfogainLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                     vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2)<< "Loss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "Loss Layer takes no output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
  << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  BlobProto blob_proto;
  ReadProtoFromBinaryFile(this->layer_param_.source(), &blob_proto);
  infogain_.FromProto(blob_proto);
  CHECK_EQ(infogain_.num(), 1);
  CHECK_EQ(infogain_.channels(), 1);
  CHECK_EQ(infogain_.height(), infogain_.width());
}

template<typename Dtype>
Dtype InfogainLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                             const bool propagate_down,
                                             vector<Blob<Dtype>*>* bottom) {
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  const Dtype* bottom_label = (*bottom)[1]->cpu_data();
  const Dtype* infogain_mat = infogain_.cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  int num = (*bottom)[0]->num();
  int dim = (*bottom)[0]->count() / (*bottom)[0]->num();
  CHECK_EQ(infogain_.height(), dim);
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    for (int j = 0; j < dim; ++j) {
      Dtype prob = max(bottom_data[i * dim + j], Dtype(kLOG_THRESHOLD));
      loss -= infogain_mat[label * dim + j] * log(prob);
      bottom_diff[i * dim + j] = -infogain_mat[label * dim + j] / prob / num;
    }
  }
  return loss / num;
}

template<typename Dtype>
void EuclideanLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2)<< "Loss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "Loss Layer takes no as output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
  << "The data and label should have the same number.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  difference_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template<typename Dtype>
Dtype EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                              const bool propagate_down,
                                              vector<Blob<Dtype>*>* bottom) {
  int count = (*bottom)[0]->count();
  int num = (*bottom)[0]->num();
  caffe_sub(count, (*bottom)[0]->cpu_data(), (*bottom)[1]->cpu_data(),
            difference_.mutable_cpu_data());
  Dtype loss = caffe_cpu_dot(count, difference_.cpu_data(),
                             difference_.cpu_data()) / num / Dtype(2);
  // Compute the gradient
  caffe_axpby(count, Dtype(1) / num, difference_.cpu_data(), Dtype(0),
              (*bottom)[0]->mutable_cpu_diff());
  return loss;
}

template<typename Dtype>
void AccuracyLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                 vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2)<< "Accuracy Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 1) << "Accuracy Layer takes 1 output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
  << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 2, 1, 1);
}

template<typename Dtype>
void AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  Dtype logprob = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  for (int i = 0; i < num; ++i) {
    // Accuracy
    Dtype maxval = -FLT_MAX;
    int max_id = 0;
    for (int j = 0; j < dim; ++j) {
      if (bottom_data[i * dim + j] > maxval) {
        maxval = bottom_data[i * dim + j];
        max_id = j;
      }
    }
    if (max_id == static_cast<int>(bottom_label[i])) {
      ++accuracy;
    }
    Dtype prob = max(bottom_data[i * dim + static_cast<int>(bottom_label[i])],
                     Dtype(kLOG_THRESHOLD));
    logprob -= log(prob);
  }
  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
  (*top)[0]->mutable_cpu_data()[1] = logprob / num;
}

template<typename Dtype>
void VerificationAccuracyLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                             vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 4)<< "VerificationAccuracyLayer takes four blobs as input.";
  CHECK_EQ(top->size(), 1) << "VerificationAccuracy Layer takes 1 output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
  << "The data and label should have the same number.";
  //CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 2, 1, 1);
  diffy_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  M_ = this->layer_param_.dual_threshold();
  LOG(INFO) << "Initial: " << M_;
}

template<typename Dtype>
void VerificationAccuracyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  Dtype logprob = 0;
  const Dtype* bottom_data1 = bottom[0]->cpu_data();
  const Dtype* bottom_label1 = bottom[1]->cpu_data();
  const Dtype* bottom_data2 = bottom[2]->cpu_data();
  const Dtype* bottom_label2 = bottom[3]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();

  int count = bottom[0]->count();
  Dtype* diffy = diffy_.mutable_cpu_data();
  caffe_sub(count, bottom_data1, bottom_data2, diffy);

  Dtype M2 = M_ * M_;
  for (int i = 0; i < num; ++i) {
    int l1 = static_cast<int>(bottom_label1[i]);
    int l2 = static_cast<int>(bottom_label2[i]);
    int offset = i * dim;
    Dtype norm2 = caffe_cpu_dot(dim, diffy + offset, diffy + offset);
    if (l1 == l2 && norm2 <= M2)
      accuracy++;
    else if (l1 != l2 && norm2 > M2)
      accuracy++;
  }
  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
  (*top)[0]->mutable_cpu_data()[1] = logprob / num;
}

template<typename Dtype>
void VerificationAccuracyLayerSiamese<Dtype>::SetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 4)<< "VerificationAccuracyLayer takes four blobs as input.";
  CHECK_EQ(top->size(), 1) << "VerificationAccuracy Layer takes 1 output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
  << "The data and label should have the same number.";
  //CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 2, 1, 1);
  diffy_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  M_ = this->layer_param_.dual_threshold();
  LOG(INFO) << "Initial: " << M_;
}

template<typename Dtype>
void VerificationAccuracyLayerSiamese<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  Dtype logprob = 0;
  const Dtype* bottom_data1 = bottom[0]->cpu_data();
  const Dtype* bottom_label1 = bottom[1]->cpu_data();
  const Dtype* bottom_data2 = bottom[2]->cpu_data();
  const Dtype* bottom_label2 = bottom[3]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();

  //LOG(INFO) << "num: " << num << ", dim: " << dim;


  int count = bottom[0]->count();
  Dtype* diffy = diffy_.mutable_cpu_data();
  caffe_sub(count, bottom_data1, bottom_data2, diffy);

  Dtype M2 = M_ * M_;
  Dtype debug_cumulative_norm=Dtype(0.);

  for (int i = 0; i < num; ++i) {
    int l1 = static_cast<int>(bottom_label1[i]);
    int l2 = static_cast<int>(bottom_label2[i]);
    int offset = i * dim;
    Dtype norm2 = caffe_cpu_dot(dim, diffy + offset, diffy + offset);
    //debug_cumulative_norm += norm2;

    std::vector<std::pair<int, int> >::const_iterator it =
        correspondance_labels_.begin();
    bool same = false;
    //  LOG(INFO) << "Looking for pair " << l1 << " " << l2;
    while (it != correspondance_labels_.end()) {
      if (it->first == l2 && it->second == l1
          || it->first == l1 && it->second == l2) {
        same = true;
        break;
      }
      it++;
    }

    //LOG(INFO) << ((same)?"positive, ":"negative, ") << l1 << ' ' << l2 << ' ' << norm2 << ' ' << M2;
    //if(same) LOG(INFO) << "Positive pair " << l1 <<", " << l2;
    //else LOG(INFO) << "Negative pair " << l1 <<", " << l2;
    if (same && norm2 <= M2) {
      //LOG(INFO) << "norm2: " << norm2 << ", M^2: " << M2;
      accuracy++;
    } else if (!same && norm2 > M2) {
      //LOG(INFO) << "norm2: " << norm2 << ", M^2: " << M2;
      accuracy++;
    }

    // LOG(INFO) << "Accuracy: " << accuracy;
    (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
    (*top)[0]->mutable_cpu_data()[1] = logprob / num;
    //LOG(INFO) << "TOTAL CUMULATIVE NORM CPU: " << debug_cumulative_norm;
  }
}

template<typename Dtype>
void VerificationAccuracyLayerSiamese<Dtype>::ReadCorrespondancesFile() {
  //XXX: should not be hardcoded
  correspondance_labels_.clear();
  const std::string &corr_file = this->layer_param_.test_net_positive_lc().c_str();
      //"/media/DATA/Datasets/SLAM_LOOP/loop_closures_positive.txt";
  LOG(INFO)<< "Reading correspondance labels for closed loops from file " << corr_file;
  int l1, l2;
  int read = 0;
  std::string line;
  std::vector<std::pair<float, float> > ssss;

  std::ifstream infile(corr_file);
  CHECK(infile.is_open()) << "Impossible to open file " << corr_file;
  std::getline(infile, line);
  while (infile >> l1 >> l2) {
    //LOG(INFO)<< "pair1: " << l1 << " " << l2;
    correspondance_labels_.push_back(std::make_pair(l1, l2));
    read++;
  }
  infile.close();
  LOG(INFO)<< "Read "<< correspondance_labels_.size() << " correspondances";
}

  INSTANTIATE_CLASS(MultinomialLogisticLossLayer);
  INSTANTIATE_CLASS(InfogainLossLayer);
  INSTANTIATE_CLASS(EuclideanLossLayer);
  INSTANTIATE_CLASS(AccuracyLayer);
  INSTANTIATE_CLASS(VerificationAccuracyLayer);
  INSTANTIATE_CLASS(VerificationAccuracyLayerSiamese);

}  // namespace caffe
