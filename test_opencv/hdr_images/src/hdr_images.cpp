// Copyright (c) 2018
// All Rights Reserved.
// Author: buggy@bcorp (BT)
#include "hdr_images.hpp"

namespace img_processing {

const std::array<float, 4>
HDRImages::timesArray = {1/30.0f, 0.25, 2.5, 15.0};

HDRImages::HDRImages(const ImageArray& images) : images(images) {}

HDRImages::~HDRImages() {}

void HDRImages::clear() { images.clear(); }

const cv::Mat& HDRImages::getHDRImage() const { return hdr_image; }

void HDRImages::setHDRImage(const cv::Mat& image) { hdr_image = image; }

void HDRImages::hdrProcessing() {
  // align images
  cv::Ptr<cv::AlignMTB> alignMTB = cv::createAlignMTB();
  alignMTB->process(images, images);

  cv::Mat responseDebevec;
  cv::Ptr<cv::CalibrateDebevec> calibrateDebevec = cv::createCalibrateDebevec();
  calibrateDebevec->process(
      images,
      responseDebevec,
      std::vector<float>(timesArray.begin(), timesArray.end()));

  cv::Mat hdrDebevec;
  cv::Ptr<cv::MergeDebevec> mergeDebevec = cv::createMergeDebevec();
  mergeDebevec->process(
      images,
      hdrDebevec,
      std::vector<float>(timesArray.begin(), timesArray.end()),
      responseDebevec);
  // set the result
  setHDRImage(hdrDebevec);
}

}  // namespace img_processing
