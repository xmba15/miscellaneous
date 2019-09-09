/**
 * @file    ColorMapping.cpp
 *
 * @author  btran
 *
 * @date    2019-09-05
 *
 * Copyright (c) organization
 *
 */

#include <string>

#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
#ifdef DATA_PATH
    const std::string IMAGE_PATH = std::string(DATA_PATH) + "/teddy/";
    const std::string FIRST = IMAGE_PATH + "im2.png";
    const std::string SECOND = IMAGE_PATH + "im6.png";

    const cv::Mat first = cv::imread(FIRST);
    const cv::Mat second = cv::imread(SECOND);

    cv::imshow("first", first);
    cv::waitKey(0);
    cv::destroyAllWindows();
#endif  // DATA_PATH

    return 0;
}
