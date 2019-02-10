// Copyright (c) 2018
// All Rights Reserved.

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include "garbor_filter.hpp"

using image_processing::generateAlignedImages;

int main(int argc, char *argv[]) {
  const std::string irisGarborImgPath =
      "../images/2018-12-17-19-10-19-1-gabor.png";

  cv::Mat irisGarborImg = cv::imread(irisGarborImgPath);
  std::vector<cv::Mat> imgs =
      generateAlignedImages(irisGarborImg);

  cv::waitKey(0);
  cv::destroyAllWindows();

  for (int i = 0; i < imgs.size(); ++i) {
    cv::Mat viz2;
    std::stringstream ss;
    ss << "image_" << i << ".jpg";
    cv::imwrite(ss.str(), imgs[i]);
  }

  return 0;
}
