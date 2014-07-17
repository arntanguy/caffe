// Copyright Arnaud TANGUY

#ifndef CAFFE_DEBUG_HPP_
#define CAFFE_DEBUG_HPP_

#include "caffe/blob.hpp"
#include "caffe/common.hpp"

#include <opencv2/opencv.hpp>

#include <ostream>

#ifndef NDEBUG_GUI
#define DEBUG_GUI 1
#else
#define DEBUG_GUI 0
#endif

/**
 * Display a batch of images from continuus data by converting it to opencv's
 * interleaved format and sticking them together
 * This is meant for debug purposes only
 **/
template <typename Dtype>
void displayImageFromData(Dtype *data, const int width, const int height, const int batchsize, const int channels)
{
  float *interleaved = new float[batchsize*channels*width*height];
  cv::Mat M(batchsize*width, height, CV_32FC3, interleaved);
  cv::namedWindow("img", cv::WINDOW_AUTOSIZE);
  for(int j=0; j<batchsize; j++) {
    float *d = reinterpret_cast<float*>(data + j * width * height * channels);
    float *ci = interleaved + j * height * width * channels;
    CHECK(d != 0) << "Null image, can't display!";
    for (int i = 0; i < width*height; ++i) {
      /**
       * Convert to OpenCV's RGB format
       */
      ci[i*channels] = d[i]/255.f;
      ci[i*channels+1] = d[i+width*height]/255.f;
      ci[i*channels+2] = d[i+2*width*height]/255.f;
    }
  }
  cv::imshow("img", M);
  cv::waitKey(0);
  delete[] interleaved;
}

template void displayImageFromData<float>(float*, const int, const int, const int, const int);
template void displayImageFromData<double>(double*, const int, const int, const int, const int);

template<typename T>
std::ostream& operator << (std::ostream &out, const caffe::Blob<T> &blob) {
  out << "\n[ num: " << blob.num() 
      <<"\n  count: " << blob.count() 
      << "\n  channels " << blob.channels()  
      << "\n  Width x Height: " << blob.width() << "x" << blob.height() << " ]\n";
  return out;
}

#endif
