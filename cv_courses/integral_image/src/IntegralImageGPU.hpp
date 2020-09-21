/**
 * @file    IntegralImageGPU.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/scan.h>

namespace gpu
{
namespace
{
struct TransposeIndex : public thrust::unary_function<int, int> {
    __host__ __device__ TransposeIndex(int height, int width) : height(height), width(width)
    {
    }

    __host__ __device__ int operator()(int idx) const
    {
        int row = idx / width;
        int col = idx % width;
        return height * col + row;
    }

    int height, width;
};

template <typename T, template <typename...> class Container>
void transposeMat(int height, int width, const Container<T>& src, Container<T>& dst)
{
    thrust::counting_iterator<int> indices(0);
    thrust::gather(thrust::make_transform_iterator(indices, TransposeIndex(width, height)),
                   thrust::make_transform_iterator(indices, TransposeIndex(width, height)) + dst.size(), src.begin(),
                   dst.begin());
}

struct whichRow : thrust::unary_function<int, int> {
    int width;

    __host__ __device__ whichRow(int width) : width(width)
    {
    }

    __host__ __device__ int operator()(int idx) const
    {
        return idx / width;
    }
};

template <typename T, template <typename...> class Container>
void scanMatrixByRows(Container<T>& u, int height, int width)
{
    thrust::counting_iterator<int> indices(0);
    thrust::transform_iterator<whichRow, thrust::counting_iterator<int>> keyIterator(indices, whichRow(width));
    thrust::inclusive_scan_by_key(keyIterator, keyIterator + height * width, u.begin(), u.begin());
}
}  // namespace

template <typename T> void createIntegralImage(const T* src, T* dst, int height, int width)
{
    thrust::device_vector<T> dDst(src, src + height * width);
    thrust::device_vector<T> dBuffer(height * width);

    scanMatrixByRows<T>(dDst, height, width);
    transposeMat<int, thrust::device_vector>(height, width, dDst, dBuffer);
    scanMatrixByRows<T>(dBuffer, width, height);
    transposeMat<int, thrust::device_vector>(width, height, dBuffer, dDst);
    thrust::copy(dDst.begin(), dDst.end(), dst);
}
}  // namespace gpu
