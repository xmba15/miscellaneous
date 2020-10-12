/**
 * @file    TensorRTHandler.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <any>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>

namespace trt
{
template <typename PType> void validate(const PType& param);

class TensorRTHandler
{
 private:
    struct TRTDestroy {
        template <class T> void operator()(T* obj) const
        {
            if (obj) {
                obj->destroy();
            }
        }
    };

 protected:
    template <class T> using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

    explicit TensorRTHandler(const std::string& modelPath, const std::size_t batchSize = 1,
                             const std::size_t maxBatchSize = 1, const std::size_t maxWorkspaceSize = 1UL << 30);
    virtual ~TensorRTHandler();

    static std::size_t getSizeByDim(const nvinfer1::Dims& dims)
    {
        return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<std::size_t>());
    }

    const auto& getEngine() const
    {
        return m_engine;
    }

 protected:
    bool allocateBuffers();

 protected:
    TRTUniquePtr<nvinfer1::IBuilder> m_builder;
    TRTUniquePtr<nvinfer1::INetworkDefinition> m_network;
    TRTUniquePtr<nvinfer1::IBuilderConfig> m_config;
    TRTUniquePtr<nvinfer1::ICudaEngine> m_engine;
    TRTUniquePtr<nvinfer1::IExecutionContext> m_context;

    std::string m_modelPath;
    std::size_t m_batchSize;
    std::size_t m_maxBatchSize;
    std::size_t m_maxWorkspaceSize;

    std::vector<void*> m_buffers;

    std::vector<nvinfer1::Dims> m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;
};

class TensorRTHandlerOnnx : public TensorRTHandler
{
 protected:
    explicit TensorRTHandlerOnnx(const std::string& modelPath, const std::size_t batchSize = 1,
                                 const std::size_t maxBatchSize = 1, const std::size_t maxWorkspaceSize = 1UL << 30);
    ~TensorRTHandlerOnnx();

 private:
    TRTUniquePtr<nvonnxparser::IParser> m_parser;
};

class ImageTensorRTHandlerOnnx : public TensorRTHandlerOnnx
{
 public:
    using AnyOutput = std::any;
    using AnyOutputs = std::vector<AnyOutput>;

    struct Param {
        int height;
        int width;
        bool normalizeImage = true;
        std::vector<double> means = {};
        std::vector<double> stds = {};
        std::string modelPath;
        std::size_t batchSize = 1;
        std::size_t maxBatchSize = 10;
        std::size_t maxWorkspaceSize = 1UL << 30;
        std::vector<std::string> classes;
    };

    const std::vector<std::string>& classes() const
    {
        return m_param.classes;
    }

 protected:
    explicit ImageTensorRTHandlerOnnx(const Param& param);
    ~ImageTensorRTHandlerOnnx();

    virtual bool preprocess(const std::vector<cv::Mat>& images) = 0;
    virtual AnyOutput postprocessOne(const std::vector<float>& curData) const = 0;
    AnyOutputs postprocess() const;
    AnyOutputs runAny(const std::vector<cv::Mat>& images);

    template <typename Container, typename Containers = std::vector<Container>>
    Containers run(const std::vector<cv::Mat>& images);

 protected:
    Param m_param;
};

template <> void validate<>(const ImageTensorRTHandlerOnnx::Param& param);

class ImageClassificationTensorRTHandlerOnnx : public ImageTensorRTHandlerOnnx
{
 public:
    using Ptr = std::shared_ptr<ImageClassificationTensorRTHandlerOnnx>;
    using Output = std::pair<int, float>;
    using Outputs = std::vector<Output>;

    explicit ImageClassificationTensorRTHandlerOnnx(const Param& param);
    ~ImageClassificationTensorRTHandlerOnnx();

    Outputs run(const std::vector<cv::Mat>& images);

 private:
    bool preprocess(const std::vector<cv::Mat>& images) final;
    AnyOutput postprocessOne(const std::vector<float>& curData) const final;
};

template <typename Container, typename Containers>
Containers ImageTensorRTHandlerOnnx::run(const std::vector<cv::Mat>& images)
{
    AnyOutputs anyOutputs = this->runAny(images);
    Containers outputs;
    outputs.reserve(anyOutputs.size());

    std::transform(anyOutputs.begin(), anyOutputs.end(), outputs.begin(),
                   [](const AnyOutput& elem) { return std::any_cast<Container>(elem); });

    for (const auto& anyOutput : anyOutputs) {
        outputs.emplace_back(std::move(std::any_cast<Container>(anyOutput)));
    }

    return outputs;
}
}  // namespace trt
