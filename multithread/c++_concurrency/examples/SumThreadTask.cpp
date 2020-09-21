/**
 * @file    SumThreadTask.cpp
 *
 * Copyright (c) organization
 *
 */

#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <numeric>
#include <vector>

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

int main(int argc, char *argv[])
{
    std::vector<int> v(40000);
    std::generate(std::begin(v), std::end(v),
                  [n = 0]() mutable { return n++; });

    auto start = std::chrono::system_clock::now();
    int sum = std::accumulate(std::begin(v), std::end(v), 0, std::plus<int>());
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "single thread: " << diff.count() * 1000 << "[ms]\n";

    auto accFunc = [](const std::vector<int> &v, int beginIdx, int endIdx) {
        return std::accumulate(std::begin(v) + beginIdx, std::begin(v) + endIdx,
                               0);
    };

    start = std::chrono::system_clock::now();
    auto t1 = std::async(accFunc, std::ref(v), 0, v.size() / 2);
    auto t2 = std::async(accFunc, std::ref(v), v.size() / 2, v.size());

    int acm1 = t1.get();
    int acm2 = t2.get();
    int sum2 = acm1 + acm2;
    end = std::chrono::system_clock::now();
    diff = end - start;
    std::cout << "two threads: " << diff.count() * 1000 << "[ms]\n";
    std::cout << sum << " " << sum2 << "\n";

    return 0;
}
