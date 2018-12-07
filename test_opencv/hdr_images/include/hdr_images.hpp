// Copyright (c) 2018
// All Rights Reserved.
#pragma once

#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>

namespace img_processing {

class HDRImages {
 public:
  typedef std::vector<cv::Mat> ImageArray;
  static const std::array<float, 4> timesArray;
  explicit HDRImages(const ImageArray&);
  const cv::Mat& getHDRImage() const;
  void setHDRImage(const cv::Mat&);
  void clear();
  ~HDRImages();

  void hdrProcessing();
 private:
  ImageArray images;
  cv::Mat hdr_image;
};

}  // namespace img_processing
