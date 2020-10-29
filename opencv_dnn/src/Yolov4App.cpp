/**
 * @file    Yolov4App.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include "Yolov4.hpp"

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: [app] [path/to/model/config] [path/to/model/weights]" << std::endl;
        return EXIT_FAILURE;
    }

    perception::Yolov4Handler::Param param;
    param.pathToModelConfig = argv[1];
    param.pathToModelWeights = argv[2];
    perception::Yolov4Handler::Ptr yolov4(new perception::Yolov4Handler(param));

    return EXIT_SUCCESS;
}
