/**
 * @file    main.cpp
 *
 * @author  btran
 *
 * @date    2020-05-03
 *
 * Copyright (c) organization
 *
 */

#include <iostream>
#include <numeric>

#include "SharedMemory.hpp"

int main(int argc, char *argv[])
{
    constexpr int N = 10;
    float hA[N];
    std::iota(hA, hA + N, 0);

    accumulateAverage(hA, N);

    for (auto elem : hA) {
        std::cout << elem << "\n";
    }

    return 0;
}
