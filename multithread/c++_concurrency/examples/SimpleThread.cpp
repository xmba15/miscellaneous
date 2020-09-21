/**
 * @file    SimpleThread.cpp
 *
 * Copyright (c) organization
 *
 */

#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

std::mutex countMtx;
std::mutex valueMtx;
std::vector<int> values;

void threadFunc(int tid)
{
    countMtx.lock();
    std::cout << tid << "\n";
    countMtx.unlock();
}

int main(int argc, char *argv[])
{
    std::vector<std::thread> threads;
    const int NUM_THREADS = 4;
    threads.reserve(NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(std::thread(threadFunc, i));
    }

    for (auto &thr : threads) {
        thr.join();
    }

    return 0;
}
