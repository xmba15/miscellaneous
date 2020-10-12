/**
 * @file    logging.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <iostream>

#include <NvInfer.h>

namespace trt
{
class Logger : public nvinfer1::ILogger
{
 public:
    explicit Logger(Severity severity = Severity::kWARNING)
    {
    }

    virtual ~Logger()
    {
    }

    nvinfer1::ILogger& getTRTLogger()
    {
        return *this;
    }

    void log(nvinfer1::ILogger::Severity severity, const char* msg) final
    {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: " << msg << std::endl;
                break;
            case Severity::kERROR:
                std::cerr << "ERROR: " << msg << std::endl;
                break;
            case Severity::kWARNING:
                std::cout << "WARNING: " << msg << std::endl;
                break;
            case Severity::kINFO:
                std::cout << "INFO: " << msg << std::endl;
                break;
            case Severity::kVERBOSE:
            default:
                break;
        }
    }
} gLogger;
}  // namespace trt
