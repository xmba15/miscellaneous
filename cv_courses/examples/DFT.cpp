/**
 * @file    DFT.cpp
 *
 * @author  btran
 *
 */

#include <opencv2/opencv.hpp>

#include <string>

int main(int argc, char* argv[])
{
    std::string IMG_PATH = argv[1];
    cv::Mat img = cv::imread(IMG_PATH, cv::IMREAD_GRAYSCALE);

    cv::Mat padded;
    int m = cv::getOptimalDFTSize(img.rows);
    int n = cv::getOptimalDFTSize(img.cols);
    cv::copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI);
    cv::split(complexI, planes);  // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

    cv::Mat idft;
    cv::dft(complexI, idft, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    cv::normalize(idft, idft, 0, 1, cv::NORM_MINMAX);

    cv::Mat inversed(idft, cv::Rect(0, 0, img.cols, img.rows));

    cv::imshow("inversed", inversed);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
