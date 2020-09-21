/**
 * @file    Example1.cpp
 *
 * @brief   Test Template MergeSort
 *
 * @author  btran
 *
 * @date    2019-06-07
 *
 * Copyright (c) organization
 *
 */

#include "MergeSort.hpp"
#include <iostream>

using MergeSort = algo::MergeSort<int, std::vector>;

int main(int argc, char *argv[])
{
    std::vector<int> oriV{1, 4, 5, 10, 45, 18, 1000, 78, 450, 123, 2000};

    std::cout << "Original Array"
              << "\n";
    std::cout << oriV << "\n";

    MergeSort ms;

    ms.sorting(oriV);

    std::cout << "Sorted Array"
              << "\n";
    std::cout << oriV << "\n";

    return 0;
}
