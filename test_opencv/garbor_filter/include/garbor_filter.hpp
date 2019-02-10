// Copyright (c) 2018
// All Rights Reserved.

#ifndef GARBOR_FILTER_HPP_
#define GARBOR_FILTER_HPP_
#include <vector>
#include <opencv2/opencv.hpp>

namespace image_processing {

/**
 * Generate multiple images to compensate the tiltness of the eyes
 *
 * @param img: single channel image of the iris region filtered by Garbor filters
 * @param regionWidth: width of the region to extract
 * @param imageNum: number of images to be generated on each side
 * @return vector of images generated to compensate the tiltness of the eyes
 */
std::vector<cv::Mat> generateAlignedImages(const cv::Mat& img,
                                           const int& regionWidth = 2,
                                           const int& imageNum = 3);

}  // image_processing
#endif /* GARBOR_FILTER_HPP_ */
