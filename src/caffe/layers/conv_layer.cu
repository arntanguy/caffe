// Copyright 2013 Yangqing Jia

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void dump(const Blob<Dtype>* b){
	const Dtype *p = b->cpu_diff();
	fprintf(stderr, "SIZE %d, %d, %d \n", b->channels(), b->height(),
			b->width());
	for(int c = 0; c < b->channels(); c++){
		fprintf(stderr, "CH %d \n", c);
		for(int h = 0; h< b->height();h++){
			for(int w = 0; w< b->width();w++)
				fprintf(stderr, "%f ", p[(c*b->height() + h) *
						b->width() + w]);
			fprintf(stderr, "\n");
		}
	}
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  if(NTILE_WIDTH_ * NTILE_HEIGHT_ <= 1){
    const Dtype* weight = this->blobs_[0]->gpu_data();
    int weight_offset = M_ * K_;
    int col_offset = K_ * N_;
    int top_offset = M_ * N_;
    for (int n = 0; n < NUM_; ++n) {
      // First, im2col
      im2col_gpu(bottom_data + bottom[0]->offset(n), CHANNELS_, HEIGHT_,
          WIDTH_, KSIZE_, PAD_, STRIDE_, col_data);
      // Second, innerproduct with groups
      for (int g = 0; g < GROUP_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
            (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
            (Dtype)0., top_data + (*top)[0]->offset(n) + top_offset * g);
      }
      // third, add bias
      if (biasterm_) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, NUM_OUTPUT_,
            N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
            reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
            (Dtype)1., top_data + (*top)[0]->offset(n));
      }
    }
  }else{
    CHECK_EQ(STRIDE_, 1);
    CHECK_EQ(PAD_, 0);
    CHECK_EQ(GROUP_, 1);
    CHECK_EQ(col_buffer_.height(), TILE_HEIGHT_);
    Dtype *out_buffer = out_buffer_.mutable_gpu_data();
    for (int n = 0; n < NUM_; ++n) {
      for(int ny = 0; ny < NTILE_HEIGHT_; ny++){
        for(int nx = 0; nx < NTILE_WIDTH_; nx++){
          int idx = ny * NTILE_WIDTH_ + nx;
          const Dtype* weight = this->blobs_[idx]->gpu_data();
          const Dtype * img = bottom_data + bottom[0]->offset(n, 0,
                TILE_HEIGHT_ * ny, TILE_WIDTH_ * nx);
          im2col_tile_gpu(img,   CHANNELS_, HEIGHT_,
              WIDTH_, KSIZE_, col_data,
              TILE_HEIGHT_, TILE_WIDTH_);
	  //dump(&col_buffer_);
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
              (Dtype)1., weight, col_data, (Dtype)0., out_buffer);
          if (biasterm_) {
            const Dtype *bias_ptr = this->blobs_[idx + NTILE_WIDTH_ *
		    NTILE_HEIGHT_]->gpu_data();
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, NUM_OUTPUT_,
                N_, 1, (Dtype)1., bias_ptr,
                reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
                (Dtype)1., out_buffer);
          }
	  //dump(&out_buffer_);
	  /* copy back */

	  int height_out = HEIGHT_ - KSIZE_ + 1;
	  int width_out = WIDTH_ - KSIZE_ + 1;
	  copy_stride_gpu(out_buffer, NUM_OUTPUT_, TILE_HEIGHT_, TILE_WIDTH_,
		top_data + (*top)[0]->offset(n, 0, TILE_HEIGHT_*ny,
			TILE_WIDTH_*nx), height_out, width_out);

        }
      }
    }/* n */
  }
}

template <typename Dtype>
Dtype ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  Dtype* col_diff = col_buffer_.mutable_gpu_diff();
  // bias gradient if necessary
  Dtype* bias_diff = NULL;

  int ntiles = NTILE_WIDTH_ * NTILE_HEIGHT_;
  if(ntiles <= 1){
	  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
	  const Dtype* top_diff = top[0]->gpu_diff();
	  const Dtype* weight = this->blobs_[0]->gpu_data();
	  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
	  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
	  if (biasterm_) {
	    bias_diff = this->blobs_[1]->mutable_gpu_diff();
	    CUDA_CHECK(cudaMemset(bias_diff, 0,
		  sizeof(Dtype) * this->blobs_[1]->count()));
	    for (int n = 0; n < NUM_; ++n) {
	      caffe_gpu_gemv<Dtype>(CblasNoTrans, NUM_OUTPUT_, N_,
		  1., top_diff + top[0]->offset(n),
		  reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
		  1., bias_diff);
	    }
	  }

	  int weight_offset = M_ * K_;
	  int col_offset = K_ * N_;
	  int top_offset = M_ * N_;
	  CUDA_CHECK(cudaMemset(weight_diff, 0,
		sizeof(Dtype) * this->blobs_[0]->count()));
	  for (int n = 0; n < NUM_; ++n) {
	    // since we saved memory in the forward pass by not storing all col data,
	    // we will need to recompute them.
	    im2col_gpu(bottom_data + (*bottom)[0]->offset(n), CHANNELS_, HEIGHT_,
		WIDTH_, KSIZE_, PAD_, STRIDE_, col_data);
	    // gradient w.r.t. weight. Note that we will accumulate diffs.
	    for (int g = 0; g < GROUP_; ++g) {
	      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
		  (Dtype)1., top_diff + top[0]->offset(n) + top_offset * g,
		  col_data + col_offset * g, (Dtype)1.,
		  weight_diff + weight_offset * g);
	    }
	    // gradient w.r.t. bottom data, if necessary
	    if (propagate_down) {
	      for (int g = 0; g < GROUP_; ++g) {
		caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
		    (Dtype)1., weight + weight_offset * g,
		    top_diff + top[0]->offset(n) + top_offset * g,
		    (Dtype)0., col_diff + col_offset * g);
	      }
	      // col2im back to the data
	      col2im_gpu(col_diff, CHANNELS_, HEIGHT_, WIDTH_, KSIZE_, PAD_, STRIDE_,
		  bottom_diff + (*bottom)[0]->offset(n));
	    }
	  }
  }else{
    CHECK_EQ(GROUP_, 1);
    Dtype *out_buffer = out_buffer_.mutable_gpu_data();
    const Dtype* bottom_data = (*bottom)[0]->gpu_data();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    for(int i = 0; i < ntiles; i++){
	    if (biasterm_) {
		    bias_diff = this->blobs_[ntiles + i]->mutable_gpu_diff();
		    CUDA_CHECK(cudaMemset(bias_diff, 0,
					    sizeof(Dtype) * this->blobs_[ntiles+i]->count()));
	    }

	    CUDA_CHECK(cudaMemset(this->blobs_[i]->mutable_gpu_diff(), 0,
				    sizeof(Dtype) * this->blobs_[i]->count()));
    }
    //XXX overlap region ??
    CUDA_CHECK(cudaMemset(bottom_diff, 0,
		    sizeof(Dtype) * (*bottom)[0]->count()));

    for (int n = 0; n < NUM_; ++n) {
	    for(int ny = 0; ny < NTILE_HEIGHT_; ny++){
		    for(int nx = 0; nx < NTILE_WIDTH_; nx++){
			    int idx = ny * NTILE_WIDTH_ + nx;
			    Dtype* weight_diff =
				    this->blobs_[idx]->mutable_gpu_diff();
			    const Dtype * img = bottom_data + (*bottom)[0]->offset(n, 0,
					    TILE_HEIGHT_ * ny, TILE_WIDTH_ * nx);
			    im2col_tile_gpu(img,   CHANNELS_, HEIGHT_,
					    WIDTH_, KSIZE_, col_data,
					    TILE_HEIGHT_, TILE_WIDTH_);

			    int height_out = HEIGHT_ - KSIZE_ + 1;
			    int width_out = WIDTH_ - KSIZE_ + 1;

			    const Dtype* top_diff = top[0]->gpu_diff();
			    copy_stride_gather_gpu(out_buffer, NUM_OUTPUT_, TILE_HEIGHT_, TILE_WIDTH_,
					    top_diff + top[0]->offset(n, 0, TILE_HEIGHT_*ny,
						    TILE_WIDTH_*nx), height_out, width_out);

			    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
					    (Dtype)1., out_buffer,
					    col_data, (Dtype)1.,
					    weight_diff);
			    if(biasterm_) {
				    bias_diff = this->blobs_[ntiles + idx]->mutable_gpu_diff();
				    caffe_gpu_gemv<Dtype>(CblasNoTrans, NUM_OUTPUT_, N_,
						    1., out_buffer,
						    reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
						    1., bias_diff);
			    }
			    if(propagate_down){
				    const Dtype* weight = this->blobs_[idx]->gpu_data();
				    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
						    (Dtype)1., weight,
						    out_buffer,
						    (Dtype)0., col_diff);
				    // col2im back to the data
				    col2im_tile_gpu(col_diff, CHANNELS_,
						    TILE_HEIGHT_, TILE_WIDTH_,
						    KSIZE_, HEIGHT_, WIDTH_,
						    bottom_diff +
						    (*bottom)[0]->offset(n,0, TILE_HEIGHT_*ny, TILE_WIDTH_*nx));
			    }
		    }
	    }
    }
  }
  return Dtype(0.);
}


INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
