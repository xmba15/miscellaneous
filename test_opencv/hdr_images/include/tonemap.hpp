// Copyright (c) 2018
// All Rights Reserved.
// Author: buggy@bcorp (BT)
#pragma once

#include <opencv2/opencv.hpp>

namespace img_processing {

template<class T>
class ToneMap {
 public:
  explicit ToneMap(const cv::Ptr<T>&);
  ~ToneMap() {}
  void processHDRImage(const cv::Mat&);
  void setMappedImage(const cv::Mat&);
  const cv::Mat& getMappedImage() const;
 private:
  cv::Mat mappedImage;
  cv::Ptr<T> tonemapEngine;
};

template<class T>
ToneMap<T>::ToneMap(const cv::Ptr<T>& t) : tonemapEngine(t) {}

template<class T>
const cv::Mat& ToneMap<T>::getMappedImage() const {
  return mappedImage;
}

template<class T>
void ToneMap<T>::processHDRImage(const cv::Mat& hdr_img) {
  tonemapEngine->process(hdr_img, mappedImage);
  mappedImage *= 3;
  mappedImage *= 255;
}

}  // namespace img_processing
