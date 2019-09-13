/**
 * @file    Utility.hpp
 *
 * @author  btran
 *
 * @date    2019-09-09
 *
 * Copyright (c) organization
 *
 */

/**
 * L*a*b
   - L: lightness from black(0) -> white(100)
   - a: green (-)  to red(+)
   - b: blue (-) to yellow(+)

 * References: - http://www.easyrgb.com/en/math.php#text2
               - http://wiki.nuaj.net/index.php/Color_Transforms
**/

#pragma once

#include <memory>
#include <opencv2/opencv.hpp>

template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
std::ostream &operator<<(std::ostream &os, const Container<T> &container)
{
    using ContainerType = Container<T>;
    for (typename ContainerType::const_iterator it = container.begin();
         it != container.end(); ++it) {
        os << *it << " ";
    }

    return os;
}

template <typename T, size_t SIZE>
std::ostream &operator<<(std::ostream &os, const std::array<T, SIZE> &container)
{
    for (auto it = container.cbegin(); it != container.cend(); ++it) {
        os << *it << " ";
    }

    return os;
}

namespace img_process
{
namespace util
{
// Observer= 2°, Illuminant= D65

std::array<double, 3> rgb2xyz(const cv::Scalar &rgb)
{
    std::array<double, 3> result;
    const uchar R = rgb[0];
    const uchar G = rgb[1];
    const uchar B = rgb[2];

    double varR = R / 255.f;
    double varG = G / 255.f;
    double varB = B / 255.f;

    varR = (varR > 0.04045) ? std::pow(((varR + 0.055) / 1.055), 2.4)
                            : varR / 12.92;
    varG = (varG > 0.04045) ? std::pow(((varG + 0.055) / 1.055), 2.4)
                            : varG / 12.92;
    varB = (varB > 0.04045) ? std::pow(((varB + 0.055) / 1.055), 2.4)
                            : varB / 12.92;

    result[0] = varR * 0.4124 + varG * 0.3576 + varB * 0.1805;
    result[1] = varR * 0.2126 + varG * 0.7152 + varB * 0.0722;
    result[2] = varR * 0.0193 + varG * 0.1192 + varB * 0.9505;

    return result;
}

cv::Scalar xyz2rgb(const std::array<double, 3> &xyz)
{
    std::array<double, 3> result;

    result[0] = xyz[0] * 3.2406 + xyz[1] * -1.5372 + xyz[2] * -0.4986;
    result[1] = xyz[0] * -0.9689 + xyz[1] * 1.8758 + xyz[2] * 0.0415;
    result[2] = xyz[0] * 0.0557 + xyz[1] * -0.2040 + xyz[2] * 1.0570;

    for (size_t i = 0; i < 3; ++i) {
        result[i] = (result[i] > 0.0031308)
                        ? 1.055 * (std::pow(result[i], (1 / 2.4))) - 0.055
                        : 12.92 * result[i];
    }

    return cv::Scalar(255 * result[0], 255 * result[1], 255 * result[2]);
}

// X in [0, 0.95047]
// Y in [0, 1.00000]
// Z in [0, 1.08883]
// L* in [0,100]
std::array<double, 3> xyz2lab(const std::array<double, 3> &xyz,
                              const std::array<double, 3> &ref = {
                                  0.95047, 1.000, 1.08883})
{
    std::array<double, 3> temp;
    for (size_t i = 0; i < 3; ++i) {
        temp[i] = xyz[i] / ref[i];

        temp[i] = (temp[i] > 0.008856) ? std::pow(temp[i], 1.f / 3)
                                       : (7.787 * temp[i]) + (16.f / 116);
    }

    std::array<double, 3> lab;
    lab[0] = (116 * temp[1]) - 16;
    lab[1] = 500 * (temp[0] - temp[1]);
    lab[2] = 200 * (temp[1] - temp[2]);

    return lab;
}

std::array<double, 3> lab2xyz(const std::array<double, 3> &lab,
                              const std::array<double, 3> &ref = {
                                  0.95047, 1.000, 1.08883})
{
    std::array<double, 3> xyz;

    xyz[1] = (lab[0] + 16) / 116;
    xyz[0] = lab[1] / 500.f + xyz[1];
    xyz[2] = xyz[1] - lab[2] / 200.f;

    for (size_t i = 0; i < 3; ++i) {
        const double powerOf3 = std::pow(xyz[i], 3);
        xyz[i] =
            (powerOf3 > 0.008856) ? powerOf3 : (xyz[i] - 16.f / 116) / 7.787;

        xyz[i] *= ref[i];
    }

    return xyz;
}

}  // namespace util
}  // namespace img_process
