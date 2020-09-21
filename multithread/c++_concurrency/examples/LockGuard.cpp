/**
 * @file    LockGuard.cpp
 *
 * Copyright (c) organization
 *
 */

#include <iostream>
#include <mutex>
#include <thread>

int counter = 0;
std::mutex counterMutex;

void worker()
{
    std::lock_guard<std::mutex> lock(counterMutex);
    if (counter == 1) {
        counter += 10;
    } else if (counter >= 10) {
        counter += 15;
    } else if (counter >= 50) {
    } else {
        ++counter;
    }

    std::cout << std::this_thread::get_id() << ": " << counter << "\n";
}

int main(int argc, char *argv[])
{
    std::cout << __func__ << ": " << counter << "\n";
    std::thread t1(worker);
    std::thread t2(worker);

    t1.join();
    t2.join();

    std::cout << __func__ << ": " << counter << "\n";
    return 0;
}
