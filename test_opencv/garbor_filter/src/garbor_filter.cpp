// Copyright (c) 2018
// All Rights Reserved.

#include <vector>
#include "garbor_filter.hpp"

namespace image_processing {

std::vector<cv::Mat> generateAlignedImages(const cv::Mat& img,
                                           const int& regionWidth,
                                           const int& imageNum) {
  std::vector<cv::Mat> imgs(2 * imageNum);
  int width = img.cols;
  int height = img.rows;

  for (int i = 1; i <= imageNum; ++i) {
    cv::Mat firstImg = cv::Mat::zeros(img.size(), img.type());
    cv::Mat secondImg = cv::Mat::zeros(img.size(), img.type());

    cv::Rect smallLeftRegion(0, 0, i * regionWidth, height);
    cv::Rect largeRightRegion(i * regionWidth, 0,
                              width - i * regionWidth, height);

    cv::Rect largeLeftRegion(0, 0, width - i * regionWidth, height);
    cv::Rect smallRightRegion(width - i * regionWidth, 0,
                              i * regionWidth, height);

    // image number (i-1)
    img(smallLeftRegion).copyTo(firstImg(smallRightRegion));
    img(largeRightRegion).copyTo(firstImg(largeLeftRegion));
    imgs[i-1] = firstImg;

    // image number (i-1) + 3
    img(largeLeftRegion).copyTo(secondImg(largeRightRegion));
    img(smallRightRegion).copyTo(secondImg(smallLeftRegion));
    imgs[(i-1)+3] = secondImg;
  }

  return imgs;
}

}  // image_processing
