// Copyright Arnaud TANGUY

#include "caffe/util/debug.hpp"
#include "caffe/common.hpp"

#include <opencv2/opencv.hpp>

using namespace cv;

void displayImageFromData(float *data, const int width, const int height)
{
  float *interleaved = new float[width*height*3];
  for (int i = 0; i < width*height; ++i) {
    interleaved[i*3] = data[i];
    interleaved[i*3+1] = data[i+width*height];
    interleaved[i*3+2] = data[i+2*width*height];
  }
  Mat M(height, width, CV_32FC3, interleaved);
  cv::namedWindow("img");
  cv::imshow("img", M);
  cv::waitKey(0);
  delete[] interleaved;
}
