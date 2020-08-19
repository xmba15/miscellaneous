/**
 * @file    StereoMatching.cpp
 *
 * @author  btran
 *
 */

#include <utility>

#include <opencv2/opencv.hpp>

namespace
{
struct CalibrationParam {
    double fx = 1936;
    double fy = 1096;
    double cx = 972.332144;
    double cy = 522.410129;
};

std::pair<CalibrationParam, CalibrationParam> getCalibParams();

double BASE_LINE = 2.322887e-01;
}  // namespace

int main(int argc, char* argv[])
{
    const std::string IMAGE_PATH = std::string(DATA_PATH) + "/";
    const std::string LEFT_PATH = IMAGE_PATH + "left.jpg";
    const std::string RIGHT_PATH = IMAGE_PATH + "right.jpg";

    const cv::Mat left = cv::imread(LEFT_PATH, 0);
    const cv::Mat right = cv::imread(RIGHT_PATH, 0);

    int mindisparity = 0;
    int ndisparities = ((left.cols / 8) + 15) & -16;
    int SADWindowSize = 11;

    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);

    cv::Mat disp;
    sgbm->compute(left, right, disp);

    //divide by 16 to get the true disparity value
    disp.convertTo(disp, CV_32F, 1.0 / 16);
    disp = disp.colRange(80, disp.cols);
    cv::Mat disp8U = cv::Mat(disp.rows, disp.cols, CV_8UC1);
    cv::normalize(disp, disp8U, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imshow("results/BM.jpg", disp8U);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

namespace
{
std::pair<CalibrationParam, CalibrationParam> getCalibParams()
{
    CalibrationParam left, right;
    left.fx = 1936;
    left.fy = 1096;
    left.cx = 972.332144;
    left.cy = 522.410129;

    right.fx = 1936;
    right.fy = 1096;
    right.cx = 932.214570;
    right.cy = 525.280171;

    return std::make_pair(left, right);
}
}  // namespace
