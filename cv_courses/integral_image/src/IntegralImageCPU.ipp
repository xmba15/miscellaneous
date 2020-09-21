/**
 * @file    IntegralImageCPU.ipp
 *
 * @author  btran
 *
 */

#include "IntegralImageCPU.hpp"

#include <iostream>

namespace cpu
{
/**
 *  \brief ii(x,y)=I(x,y)+ii(x-1,y)+ii(x,y-1)-ii(x-1,y-1)
 */
template <typename T> void createIntegralImageNaive(const T* src, T* dst, int height, int width)
{
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            dst[i * width + j] = src[i * width + j];

            if (i - 1 >= 0 && j - 1 >= 0) {
                dst[i * width + j] -= dst[(i - 1) * width + (j - 1)];
            }

            if (i - 1 >= 0) {
                dst[i * width + j] += dst[(i - 1) * width + j];
            }

            if (j - 1 >= 0) {
                dst[i * width + j] += dst[i * width + j - 1];
            }
        }
    }
}

template <typename T> void createIntegralImageCumulativeRow(const T* src, T* dst, int height, int width)
{
    for (int i = 0; i < height; ++i) {
        T rowSum = 0;
        for (int j = 0; j < width; ++j) {
            rowSum += src[i * width + j];
            dst[i * width + j] = rowSum;
            if (i - 1 < 0) {
                continue;
            }
            dst[i * width + j] += dst[(i - 1) * width + j];
        }
    }
}
}  // namespace cpu
