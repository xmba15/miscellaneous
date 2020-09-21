/**
 * @file    Timer.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <chrono>

class TimerBase
{
 public:
    TimerBase();
    void clear();
    bool isStarted() const;
    void start();
    const uint64_t getMs() const;

 private:
    std::chrono::system_clock::time_point _start;
};

TimerBase::TimerBase()
    : _start(std::chrono::system_clock::time_point::min())
{
}

void TimerBase::clear()
{
    this->_start = std::chrono::system_clock::time_point::min();
}

bool TimerBase::isStarted() const
{
    return (this->_start.time_since_epoch() != std::chrono::system_clock::duration(0));
}

void TimerBase::start()
{
    this->_start = std::chrono::system_clock::now();
}

const uint64_t TimerBase::getMs() const
{
    if (this->isStarted()) {
        const std::chrono::system_clock::duration diff = std::chrono::system_clock::now() - this->_start;
        return (unsigned)(std::chrono::duration_cast<std::chrono::milliseconds>(diff).count());
    }
    return 0;
}
