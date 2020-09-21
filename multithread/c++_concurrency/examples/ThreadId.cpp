/**
 * @file    ThreadId.cpp
 *
 * Copyright (c) organization
 *
 */

#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

std::mutex displayMutex;

void worker()
{
    std::thread::id thisId = std::this_thread::get_id();
    displayMutex.lock();
    std::cout << "thread " << thisId << " sleeping...\n";
    displayMutex.unlock();
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

int main(int argc, char *argv[])
{
    std::thread t1(worker);
    std::thread::id t1Id = t1.get_id();

    std::thread t2(worker);
    std::thread::id t2Id = t2.get_id();

    displayMutex.lock();
    std::cout << "t1's id: " << t1Id << "\n";
    std::cout << "t2's id: " << t2Id << "\n";
    displayMutex.unlock();

    t1.join();
    t2.join();
    return 0;
}
