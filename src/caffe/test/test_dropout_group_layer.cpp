// Copyright 2013 Yangqing Jia

#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class DropoutGroupLayerTest : public ::testing::Test {
 protected:
  DropoutGroupLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.0);
    ConstantFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~DropoutGroupLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(DropoutGroupLayerTest, Dtypes);

TYPED_TEST(DropoutGroupLayerTest, TestSetup) {
  LayerParameter layer_param;
  DropoutGroupLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(DropoutGroupLayerTest, TestGPU) {
  LayerParameter layer_param;
  //drop 3
  const double ratio = 0.2;
  layer_param.set_dropout_ratio(ratio);
  DropoutGroupLayer<TypeParam> layer(layer_param);
  Caffe::set_mode(Caffe::GPU);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.UpdateMask();
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  const TypeParam v = 1. / (1-ratio);
  for (int n = 0; n < 2; n++){
	 // fprintf(stderr, "NUM %d\n", n);
	for(int c = 0; c < 3; c++){
	  	//fprintf(stderr, "cn %d\n", c);
  		int zeros = 0;
		TypeParam sum = 0.;
		const TypeParam *p = this->blob_top_->cpu_data()
			+ this->blob_top_->offset(n, c);
		for(int i = 0; i < 5 * 6; i++){
			if(p[i] == 0)
				zeros++;
			else{
				EXPECT_LE(p[i], TypeParam(v + 1e-4));
				EXPECT_GE(p[i], TypeParam(v - 1e-4));
			}
	  		//fprintf(stderr, "%f ", p[i]);
		}
	  	//fprintf(stderr, "\n");
  		EXPECT_EQ(zeros, (int)(ratio * 30));
	}
  }
}


TYPED_TEST(DropoutGroupLayerTest, TestGPUGradient) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  DropoutGroupLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(DropoutGroupLayerTest, TestGPUUpscale) {
  LayerParameter layer_param;
  const double ratio = 0.5;
  layer_param.set_dropout_ratio(ratio);
  Caffe::set_mode(Caffe::GPU);
  DropoutGroupLayer<TypeParam> layer(layer_param);
  DropoutGroupLayer<TypeParam> up_layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.UpdateMask();
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));

  vector<Blob<TypeParam>*> blob_up_bottom_vec;
  vector<Blob<TypeParam>*> blob_up_top_vec;
  Blob<TypeParam> blob_up_bot(2,4,7,6);
  Blob<TypeParam> blob_up_top;
  blob_up_bottom_vec.push_back(&blob_up_bot);
  blob_up_top_vec.push_back(&blob_up_top);
  up_layer.SetUp(blob_up_bottom_vec, &blob_up_top_vec);

  FillerParameter filler_param;
  filler_param.set_value(1.0);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(&blob_up_bot);

  up_layer.UpscaleMaskFrom(&layer);
  up_layer.Forward(blob_up_bottom_vec, &(blob_up_top_vec));

  for (int n = 0; n < 2; n++){
	 // fprintf(stderr, "NUM %d\n", n);
	for(int c = 0; c < 3; c++){
	  	//fprintf(stderr, "cn %d\n", c);
  		int zeros = 0;
		TypeParam sum = 0.;
		const TypeParam *p = blob_up_top.cpu_data()
			+ blob_up_top.offset(n, c);
		for(int i = 0; i < 6 * 7; i++){
			if(p[i] == 0)
				zeros++;
	  		//fprintf(stderr, "%f ", p[i]);
		}
	  	fprintf(stderr, "r %f\n", zeros / (float)42);
  		//EXPECT_EQ(zeros, (int)(ratio * 30));
	}
  }
}


}  // namespace caffe
