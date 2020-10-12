/**
 * @file    main.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include "TensorRTHandler.hpp"

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "[app] [path/to/onnx/model] [path/to/image]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_PATH = argv[1];
    const std::string IMAGE_PATH = argv[2];

    trt::ImageTensorRTHandlerOnnx::Param param;
    param.modelPath = ONNX_PATH;
    param.batchSize = 3;
    param.classes = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

    trt::ImageClassificationTensorRTHandlerOnnx::Ptr trtHandler(new trt::ImageClassificationTensorRTHandlerOnnx(param));

    cv::Mat dummy = cv::imread(IMAGE_PATH);

    std::vector<cv::Mat> images{dummy, dummy, dummy};
    auto outputs = trtHandler->run(images);

    std::cout << "output size: " << outputs.size() << "\n";

    for (const auto output : outputs) {
        std::cout << output.first << " " << output.second << "\n";
    }

    return EXIT_SUCCESS;
}
