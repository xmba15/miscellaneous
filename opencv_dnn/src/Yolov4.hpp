/**
 * @file    Yolov4.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/dnn.hpp>

#include <memory>
#include <string>
#include <vector>

namespace perception
{
class Yolov4Handler
{
 public:
    struct Param {
        float confidenceThresh = 0.5;
        float nmsThresh = 0.4;
        int gpuIdx = 0;
        std::string pathToModelConfig = "";
        std::string pathToModelWeights = "";
        std::string classes;
    };

    using Ptr = std::shared_ptr<Yolov4Handler>;

 public:
    explicit Yolov4Handler(const Param& param);

 private:
    Param m_param;
    std::vector<std::string> m_classes;
    std::unique_ptr<cv::dnn::Net> m_net;
};
}  // namespace perception
