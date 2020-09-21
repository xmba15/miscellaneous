/**
 * @file    Yolov4.cpp
 *
 * @author  btran
 *
 */

#include <opencv2/dnn/all_layers.hpp>

#include "Utility.hpp"
#include "Yolov4.hpp"

namespace perception
{
Yolov4Handler::Yolov4Handler(const Param& param)
    : m_param(param)
    , m_net(nullptr)
{
    m_net.reset(new cv::dnn::Net(cv::dnn::readNetFromDarknet(m_param.pathToModelConfig, m_param.pathToModelWeights)));
    m_classes = util::split(m_param.classes, ',');
    if (m_param.gpuIdx < 0) {
        m_net->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        m_net->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    } else {
        m_net->setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        m_net->setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
}
}  // namespace perception
