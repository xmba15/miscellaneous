/**
 * @file    TestContainer.cpp
 *
 * @brief   Test Boost Container
 *
 * @author  bt
 *
 * @date    2019-06-17
 *
 * Copyright (c) organization
 *
 */

#include <boost/container/flat_map.hpp>
#include <chrono>  // NOLINT
#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>

using multimap = boost::container::flat_multimap<int, int>;

int main(int argc, char *argv[])
{
    multimap boostMap;
    std::multimap<int, int> stlMap;

    const int numIter = 100000;

    std::vector<int> firstV, secondV;
    firstV.reserve(numIter);
    secondV.reserve(numIter);

    for (int i = 0; i < numIter; ++i) {
        firstV.emplace_back(rand() % numIter);   // NOLINT
        secondV.emplace_back(rand() % numIter);  // NOLINT
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (auto first = firstV.begin(), second = secondV.begin();
         first != firstV.end() && second != secondV.end(); ++first, ++second) {
        boostMap.emplace(*first, *second);
    }

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration1 =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    start = std::chrono::high_resolution_clock::now();

    for (auto first = firstV.begin(), second = secondV.begin();
         first != firstV.end() && second != secondV.end(); ++first, ++second) {
        stlMap.emplace(*first, *second);
    }

    stop = std::chrono::high_resolution_clock::now();

    auto duration2 =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "boost flat multimap: " << duration1.count() << "[milisec]"
              << "\n";
    std::cout << "stl multimap: " << duration2.count() << "[milisec]"
              << "\n";

    return 0;
}
