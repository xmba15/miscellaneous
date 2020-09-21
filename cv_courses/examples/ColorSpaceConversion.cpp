/**
 * @file    ColorSpaceConversion.cpp
 *
 * @author  btran
 *
 * @date    2019-09-09
 *
 * Copyright (c) organization
 *
 */

#include "Utility.hpp"

int main(int argc, char *argv[])
{
    cv::Scalar color(255, 0, 0);

    auto xyz = img_process::util::rgb2xyz(color);

    std::cout << img_process::util::xyz2lab(xyz) << "\n";

    return 0;
}
