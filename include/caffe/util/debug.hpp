// Copyright Arnaud TANGUY

#ifndef CAFFE_DEBUG_HPP_
#define CAFFE_DEBUG_HPP_

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include <ostream>

void displayImageFromData(float *data, const int width, const int height);

template<typename T>
std::ostream& operator << (std::ostream &out, const caffe::Blob<T> &blob) {
  out << "\n[ num: " << blob.num() 
      <<"\n  count: " << blob.count() 
      << "\n  channels " << blob.channels()  
      << "\n  Width x Height: " << blob.width() << "x" << blob.height() << " ]\n";
  return out;
}

#endif
