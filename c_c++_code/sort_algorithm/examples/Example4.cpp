/**
 * @file    Example4.cpp
 *
 * @brief   Test Template BucketSort
 *
 * @author  btran
 *
 * @date    2019-06-10
 *
 * Copyright (c) organization
 *
 */

#include "BucketSort.hpp"
#include <iostream>

using BucketSort = algo::BucketSort<int, std::vector>;

int main(int argc, char *argv[])
{
    std::vector<int> oriV{1, 4, 5, 10, 45, 18, 1000, 78, 450, 123, 2000};

    std::cout << "Original Array"
              << "\n";
    std::cout << oriV << "\n";

    BucketSort bs;

    bs.sorting(oriV, 3);

    std::cout << "Sorted Array"
              << "\n";
    std::cout << oriV << "\n";

    return 0;
}
