/**
 * @file    ThreadMutex.cpp
 *
 * Copyright (c) organization
 *
 */

#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
std::chrono::milliseconds interval(50);
std::mutex mutex;
int sharedCounter = 0;
int exclusiveCounter = 0;
void worker0()
{
    std::this_thread::sleep_for(interval);
    while (true) {
        if (mutex.try_lock()) {
            std::cout << "Shared (" << sharedCounter << ")\n";
            mutex.unlock();
            return;
        } else {
            ++exclusiveCounter;
            std::cout << "Exclusive (" << exclusiveCounter << ")\n";
            std::this_thread::sleep_for(interval);
        }
    }
}
void worker1()
{
    mutex.lock();
    std::this_thread::sleep_for(10 * interval);
    ++sharedCounter;
    mutex.unlock();
}
int main()
{
    std::thread t1(worker0);
    std::thread t2(worker1);
    t1.join();
    t2.join();
}
