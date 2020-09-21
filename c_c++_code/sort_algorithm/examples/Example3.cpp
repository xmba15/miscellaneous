/**
 * @file    Example3.cpp
 *
 * @brief   Test Template QuickSort
 *
 * @author  btran
 *
 * @date    2019-06-10
 *
 * Copyright (c) organization
 *
 */

#include "QuickSort.hpp"
#include <iostream>

using QuickSort = algo::QuickSort<int, std::vector>;

int main(int argc, char *argv[])
{
    std::vector<int> oriV{1, 4, 5, 10, 45, 18, 1000, 78, 450, 123, 2000};

    std::cout << "Original Array"
              << "\n";
    std::cout << oriV << "\n";

    QuickSort qs;

    qs.sorting(oriV);

    std::cout << "Sorted Array"
              << "\n";
    std::cout << oriV << "\n";

    return 0;
}
