// Copyright (c) 2018
// All Rights Reserved.
// Author: buggy@bcorp (BT)
#include <iostream>
#include <string>
#include "tonemap.hpp"
#include "hdr_images.hpp"

using img_processing::HDRImages;
using img_processing::ToneMap;

int main(int argc, char *argv[]) {
  std::string image_path = "../images/";
  std::vector<std::string> filenames
      { "img_0.033.JPG", "img_0.25.JPG", "img_2.5.JPG", "img_15.JPG" };
  HDRImages::ImageArray images;
  for (auto str : filenames) {
    cv::Mat im = cv::imread(image_path + str);
    images.push_back(im);
  }

  HDRImages hdrimage(images);
  hdrimage.hdrProcessing();

  cv::Ptr<cv::TonemapMantiuk> tonemapMantiuk =
      cv::createTonemapMantiuk(2.2, 0.85, 1.2);
  ToneMap<cv::TonemapMantiuk> tonemap(tonemapMantiuk);
  tonemap.processHDRImage(hdrimage.getHDRImage());

  std::string window_name = "hdr images";
  cv::imwrite(image_path + "result.jpg", tonemap.getMappedImage());

  return 0;
}
