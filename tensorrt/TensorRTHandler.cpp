/**
 * @file    TensorRTHandler.cpp
 *
 * @author  btran
 *
 */

#include <opencv2/core/cuda.hpp>

#include <cuda_runtime_api.h>

#include "Logging.hpp"
#include "TensorRTHandler.hpp"
#include "Utility.hpp"

namespace
{
inline void CUDA_CHECK(cudaError_t ret, std::ostream& err = std::cerr)
{
    if (ret != cudaSuccess) {
        err << "Cuda failure: " << cudaGetErrorString(ret) << std::endl;
        std::abort();
    }
}
}  // namespace

namespace trt
{
TensorRTHandler::TensorRTHandler(const std::string& modelPath, const std::size_t batchSize,
                                 const std::size_t maxBatchSize, const std::size_t maxWorkspaceSize)
    : m_builder(nullptr)
    , m_network(nullptr)
    , m_config(nullptr)
    , m_engine(nullptr)
    , m_context(nullptr)
    , m_modelPath(modelPath)
    , m_batchSize(batchSize)
    , m_maxBatchSize(maxBatchSize)
    , m_maxWorkspaceSize(maxWorkspaceSize)
{
    if (m_batchSize > m_maxBatchSize) {
        throw std::runtime_error("can use batch size less than maximum value only");
    }
}

TensorRTHandler::~TensorRTHandler()
{
    for (void* buffer : m_buffers) {
        if (buffer) {
            CUDA_CHECK(cudaFree(buffer));
        }
    }
}

bool TensorRTHandler::allocateBuffers()
{
    if (!m_engine) {
        std::cerr << "engine not initialized yet" << std::endl;
        return false;
    }

    m_buffers.resize(m_engine->getNbBindings());
    for (std::size_t i = 0; i < m_engine->getNbBindings(); ++i) {
        std::size_t bindingSize =
            TensorRTHandler::getSizeByDim(m_engine->getBindingDimensions(i)) * m_batchSize * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&m_buffers[i], bindingSize));
        if (m_engine->bindingIsInput(i)) {
            m_inputDims.emplace_back(m_engine->getBindingDimensions(i));
        } else {
            m_outputDims.emplace_back(m_engine->getBindingDimensions(i));
        }
    }

    if (m_inputDims.empty() || m_outputDims.empty()) {
        std::cerr << "Expect at least one input and one output for network" << std::endl;
        return false;
    }

    return true;
}

TensorRTHandlerOnnx::TensorRTHandlerOnnx(const std::string& modelPath, const std::size_t batchSize,
                                         const std::size_t maxBatchSize, const std::size_t maxWorkspaceSize)
    : TensorRTHandler(modelPath, batchSize, maxBatchSize, maxWorkspaceSize)
    , m_parser(nullptr)
{
    m_builder.reset(nvinfer1::createInferBuilder(gLogger));

    const auto explicitBatch =
        1U << static_cast<std::uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    m_network.reset(m_builder->createNetworkV2(explicitBatch));

    m_config.reset(m_builder->createBuilderConfig());

    m_parser.reset(nvonnxparser::createParser(*m_network, gLogger));

    if (!m_parser->parseFromFile(m_modelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        throw std::runtime_error("ERROR: could not parse the model: " + m_modelPath);
    }

    m_config->setMaxWorkspaceSize(m_maxWorkspaceSize);
    m_config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    m_config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    if (m_builder->platformHasFastFp16()) {
        m_config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    m_builder->setMaxBatchSize(m_maxBatchSize);
    m_engine.reset(m_builder->buildEngineWithConfig(*m_network, *m_config));
    m_context.reset(m_engine->createExecutionContext());

    this->allocateBuffers();
}

TensorRTHandlerOnnx::~TensorRTHandlerOnnx()
{
}

template <> void validate<>(const ImageTensorRTHandlerOnnx::Param& param)
{
    if (param.classes.empty()) {
        throw std::runtime_error("empty class names");
    }
}

ImageTensorRTHandlerOnnx::ImageTensorRTHandlerOnnx(const Param& param)
    : m_param(param)
    , TensorRTHandlerOnnx(param.modelPath, param.batchSize, param.maxBatchSize, param.maxWorkspaceSize)
{
    validate<>(m_param);
}

ImageTensorRTHandlerOnnx::~ImageTensorRTHandlerOnnx()
{
}

ImageTensorRTHandlerOnnx::AnyOutputs ImageTensorRTHandlerOnnx::runAny(const std::vector<cv::Mat>& images)
{
    this->preprocess(images);

    m_context->enqueue(m_param.batchSize, m_buffers.data(), 0, nullptr);

    return this->postprocess();
}

ImageTensorRTHandlerOnnx::AnyOutputs ImageTensorRTHandlerOnnx::postprocess() const
{
    ImageClassificationTensorRTHandlerOnnx::AnyOutputs result;

    std::vector<float> cpuRawAnyOutput(this->getSizeByDim(m_outputDims[0]) * m_param.batchSize);
    CUDA_CHECK(cudaMemcpy(cpuRawAnyOutput.data(), m_buffers[1], cpuRawAnyOutput.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::vector<std::vector<float>> cpuAnyOutput;
    cpuAnyOutput.reserve(m_param.batchSize);

    for (std::size_t i = 0; i < m_param.batchSize; ++i) {
        result.emplace_back(this->postprocessOne(
            std::vector<float>(cpuRawAnyOutput.begin() + i * this->getSizeByDim(m_outputDims[0]),
                               cpuRawAnyOutput.begin() + (i + 1) * this->getSizeByDim(m_outputDims[0]))));
    }

    return result;
}

ImageClassificationTensorRTHandlerOnnx::ImageClassificationTensorRTHandlerOnnx(const Param& param)
    : ImageTensorRTHandlerOnnx(param)
{
}

ImageClassificationTensorRTHandlerOnnx::~ImageClassificationTensorRTHandlerOnnx()
{
}

bool ImageClassificationTensorRTHandlerOnnx::preprocess(const std::vector<cv::Mat>& images)
{
    if (images.empty()) {
        std::cerr << "empty data" << std::endl;
        return false;
    }

    std::size_t width = m_inputDims[0].d[3];
    std::size_t height = m_inputDims[0].d[2];
    std::size_t channels = m_inputDims[0].d[1];
    cv::Size inputSize(width, height);

    std::vector<float> allImgData(width * height * channels * m_param.batchSize);

    for (std::size_t i = 0; i < images.size(); ++i) {
        const auto& curImage = images[i];
        if (curImage.empty()) {
            std::cerr << "image " << i << "th empty" << std::endl;
            return false;
        }

        cv::Mat gray, resized;
        cv::cvtColor(curImage, gray, cv::COLOR_BGR2GRAY);
        cv::resize(gray, resized, inputSize, 0, 0, cv::INTER_NEAREST);

        if (m_param.normalizeImage) {
            resized.convertTo(resized, CV_32FC1, 1.f / 255.f);
        } else {
            resized.convertTo(resized, CV_32FC1);
        }

        for (int row = 0; row < height; ++row) {
            auto curPtr = resized.ptr<float>(row);
            for (int col = 0; col < width; ++col) {
                allImgData[i * width * height + row * width + col] = curPtr[col];
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(m_buffers[0], allImgData.data(), allImgData.size() * sizeof(float), cudaMemcpyHostToDevice));

    return true;
}

ImageClassificationTensorRTHandlerOnnx::AnyOutput
ImageClassificationTensorRTHandlerOnnx::postprocessOne(const std::vector<float>& curData) const
{
    std::vector<float> processed = curData;
    utils::softmax<float>(processed.data(), curData.size());
    auto maxElem = std::max_element(processed.begin(), processed.end());

    AnyOutput curAnyOutput = std::make_pair(static_cast<int>(std::distance(processed.begin(), maxElem)), *maxElem);

    return curAnyOutput;
}

ImageClassificationTensorRTHandlerOnnx::Outputs
ImageClassificationTensorRTHandlerOnnx::run(const std::vector<cv::Mat>& images)
{
    return ImageTensorRTHandlerOnnx::run<Output, Outputs>(images);
}
}  // namespace trt
