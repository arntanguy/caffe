// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_LOSS_LAYERS_HPP_
#define CAFFE_LOSS_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "leveldb/db.h"
#include "pthread.h"
#include "boost/scoped_ptr.hpp"
#include "hdf5.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

const float kLOG_THRESHOLD = 1e-20;

/* AccuracyLayer
  Note: not an actual loss layer! Does not implement backwards step.
  Computes the accuracy of argmax(a) with respect to b.
*/
template <typename Dtype>
class AccuracyLayer : public Layer<Dtype> {
 public:
  explicit AccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_ACCURACY;
  }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
    NOT_IMPLEMENTED;
  }

  int top_k_;
};

/* LossLayer
  Takes two inputs of same num (a and b), and has no output.
  The gradient is propagated to a.
*/
template <typename Dtype>
class LossLayer : public Layer<Dtype> {
 public:
  explicit LossLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void SetUp(
      const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top);
  virtual void FurtherSetUp(
      const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {}

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 1; }
  // We usually cannot backpropagate to the labels; ignore force_backward for
  // these inputs.
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }
};

/* EuclideanLossLayer
  Compute the L_2 distance between the two inputs.

  loss = (1/2 \sum_i (a_i - b_i)^2)
  a' = 1/I (a - b)
*/
template <typename Dtype>
class EuclideanLossLayer : public LossLayer<Dtype> {
 public:
  explicit EuclideanLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void FurtherSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_EUCLIDEAN_LOSS;
  }
  // Unlike most loss layers, in the EuclideanLossLayer we can backpropagate
  // to both inputs.
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> diff_;
};

/* HingeLossLayer
*/
template <typename Dtype>
class HingeLossLayer : public LossLayer<Dtype> {
 public:
  explicit HingeLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_HINGE_LOSS;
  }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
};

/* InfogainLossLayer
*/
template <typename Dtype>
class InfogainLossLayer : public LossLayer<Dtype> {
 public:
  explicit InfogainLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), infogain_() {}
  virtual void FurtherSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_INFOGAIN_LOSS;
  }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> infogain_;
};

/* MultinomialLogisticLossLayer
*/
template <typename Dtype>
class MultinomialLogisticLossLayer : public LossLayer<Dtype> {
 public:
  explicit MultinomialLogisticLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void FurtherSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_MULTINOMIAL_LOGISTIC_LOSS;
  }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
};

/* SigmoidCrossEntropyLossLayer
*/
template <typename Dtype>
class SigmoidCrossEntropyLossLayer : public LossLayer<Dtype> {
 public:
  explicit SigmoidCrossEntropyLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param),
          sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
          sigmoid_output_(new Blob<Dtype>()) {}
  virtual void FurtherSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_SIGMOID_CROSS_ENTROPY_LOSS;
  }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  // sigmoid_output stores the output of the sigmoid layer.
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  // Vector holders to call the underlying sigmoid layer forward and backward.
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  vector<Blob<Dtype>*> sigmoid_top_vec_;
};

// Forward declare SoftmaxLayer for use in SoftmaxWithLossLayer.
template <typename Dtype> class SoftmaxLayer;

/* SoftmaxWithLossLayer
  Implements softmax and computes the loss.

  It is preferred over separate softmax + multinomiallogisticloss
  layers due to more numerically stable gradients.

  In test, this layer could be replaced by simple softmax layer.
*/
template <typename Dtype>
class SoftmaxWithLossLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxWithLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param), softmax_layer_(new SoftmaxLayer<Dtype>(param)) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_SOFTMAX_LOSS;
  }
  virtual inline int MaxTopBlobs() const { return 2; }
  // We cannot backpropagate to the labels; ignore force_backward for these
  // inputs.
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
  // prob stores the output probability of the layer.
  Blob<Dtype> prob_;
  // Vector holders to call the underlying softmax layer forward and backward.
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  vector<Blob<Dtype>*> softmax_top_vec_;
};


template <typename Dtype>
class SiameseLossLayer : public Layer<Dtype> {
 public:
  explicit SiameseLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                     vector<Blob<Dtype>*>* top);
  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_SIAMESE_LOSS;
  }
 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> diff_;
  Dtype Ew;
  Dtype Q;
  Dtype *dd;
};

/* SiameseAccuracyLayer
  Note: not an actual loss layer! Does not implement backwards step.
  Computes the accuracy of argmax(a) with respect to b.
*/
template <typename Dtype>
class SiameseAccuracyLayer : public Layer<Dtype> {
 public:
  explicit SiameseAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_SIAMESE_ACCURACY;
  }

  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
    NOT_IMPLEMENTED;
  }
};

template <typename Dtype>
class VerificationLossLayer : public Layer<Dtype> {
 public:
  explicit VerificationLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
                     vector<Blob<Dtype>*>* top);

  void SetThreshold(Dtype t) { M_ = t; }
  Dtype GetThreshold() { return M_ ; }

  void GetMeanDistance(vector<Dtype> &dists){
    dists.clear();
    Dtype avg[2] = {0.};
    int cnt[2] = {0};
    CHECK_EQ(distance_.size(), same_.size());
    for(size_t i=0;i<distance_.size();i++){
      int s = same_[i];
      avg[s] += distance_[i];
      cnt[s] ++;
    }

    for(int i=0;i<2;i++)
      dists.push_back(avg[i] / std::max(cnt[i], 1));
  }
  void ResetDistanceStat(){
    distance_.clear();
    same_.clear();
  }
  Dtype CalcThreshold(bool update);
 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                            vector<Blob<Dtype>*>* top);
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                            vector<Blob<Dtype>*>* top);
  // Backward functions: compute the gradients for any parameters and
  // for the bottom blobs if propagate_down is true.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> diffy1_;
  Blob<Dtype> diffy2_;

  Dtype M_;
  Dtype LAMDA_;

  std::vector<Dtype> distance_;
  std::vector<int> same_;
};



}  // namespace caffe

#endif  // CAFFE_LOSS_LAYERS_HPP_
