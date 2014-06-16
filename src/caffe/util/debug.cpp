// Copyright Arnaud TANGUY

#include "caffe/util/debug.hpp"
#include "caffe/common.hpp"

#include <opencv2/opencv.hpp>

using namespace cv;

void displayImageFromData(float *data, const int width, const int height)
{
  float *interleaved = new float[width*height*3];
  for (int i = 0; i < width*height; ++i) {
    /**
     * Convert to OpenCV's RGB format
     */
    interleaved[i*3] = data[i]/255.f;
    interleaved[i*3+1] = data[i+width*height]/255.f;
    interleaved[i*3+2] = data[i+2*width*height]/255.f;
  }
  Mat M(width, height, CV_32FC3, interleaved);
  cv::namedWindow("img");
  cv::imshow("img", M);
  cv::imwrite("/tmp/img.png", M);
  cv::waitKey(0);
  delete[] interleaved;
}
