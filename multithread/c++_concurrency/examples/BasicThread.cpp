/**
 * @file    BasicThread.cpp
 *
 * @author  bt
 *
 * Copyright (c) organization
 *
 */

#include <iostream>
#include <string>
#include <thread>

void worker(int n, std::string t)
{
    std::cout << n << ": " << t << "\n";
}

int main(int argc, char *argv[])
{
    std::thread t0(worker, 1, "Test");
    std::thread t1(std::move(t0));

    t1.join();

    return 0;
}
