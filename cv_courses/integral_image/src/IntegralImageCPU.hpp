/**
 * @file    IntegralImageCPU.hpp
 *
 * @author  btran
 *
 */

#pragma once

namespace cpu
{
/**
 *  \brief ii(x,y)=I(x,y)+ii(x-1,y)+ii(x,y-1)-ii(x-1,y-1)
 */
template <typename T> void createIntegralImageNaive(const T* src, T* dst, int height, int width);

template <typename T> void createIntegralImageCumulativeRow(const T* src, T* dst, int height, int width);
}  // namespace cpu

#include "IntegralImageCPU.ipp"
