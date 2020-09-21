/**
 * @file    App.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include "BALParser.hpp"
#include <opencv2/viz.hpp>

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: [app] [path/to/data]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string dataPath = argv[1];
    _cv::BALParser dataParser(dataPath);

    cv::viz::Viz3d window("depth map");
    cv::viz::WCloud wcloud(dataParser.point3Ds);
    window.showWidget("depth map", wcloud);
    window.spin();

    return EXIT_SUCCESS;
}
