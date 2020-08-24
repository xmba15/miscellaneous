/**
 * @file    DisparityVis.cpp
 *
 * @author  btran
 *
 */

#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: [app] [disparity/image]\n";
        return EXIT_FAILURE;
    }
    const std::string IMAGE_PATH = std::string(DATA_PATH) + "/";
    const std::string DISPARITY_PATH = IMAGE_PATH + argv[1];

    cv::Mat dispMat = cv::imread(DISPARITY_PATH, 0);
    cv::applyColorMap(dispMat, dispMat, cv::COLORMAP_HSV);
    cv::imshow("results/BM.jpg", dispMat);
    cv::imwrite("disparity.jpg", dispMat);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}
