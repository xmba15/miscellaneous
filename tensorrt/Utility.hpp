/**
 * @file    Utility.hpp
 *
 * @author  btran
 *
 */

#include <algorithm>
#include <cmath>
#include <numeric>

namespace utils
{
template <typename T> inline void softmax(T* input, const size_t inputLen)
{
    const T maxVal = *std::max_element(input, input + inputLen);

    const T sum = std::accumulate(input, input + inputLen, 0.0,
                                  [&](T a, const T b) { return std::move(a) + std::exp(b - maxVal); });

    const T offset = maxVal + std::log(sum);
    for (auto it = input; it != (input + inputLen); ++it) {
        *it = std::exp(*it - offset);
    }
}
}  // namespace utils
