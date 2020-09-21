/**
 * @file    Plot.cpp
 *
 * @author  btran
 *
 */

#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/plot.hpp>

int main(int argc, char* argv[])
{
    std::vector<double> sine;
    for (int t = 0; t < 360; t++) {
        sine.push_back(std::sin(t * CV_PI / 180.0));
    }

    cv::Mat data(sine);

    cv::Ptr<cv::plot::Plot2d> plot;

#if CV_MAJOR_VERSION < 4
    plot = cv::plot::createPlot2d(data);
#else
    plot = cv::plot::Plot2d::create(data);
#endif

    plot->setPlotBackgroundColor(cv::Scalar(255, 200, 200));
    plot->setPlotLineColor(cv::Scalar(255, 0, 0));
    plot->setPlotGridColor(cv::Scalar(255, 0, 255));

    plot->setShowText(true);
    plot->setInvertOrientation(true);
    plot->setNeedPlotLine(true);

    while (true) {
        double value = *sine.begin();
        sine.erase(sine.begin());
        sine.push_back(value);

        cv::Mat image;
        plot->render(image);
        cv::imshow("sine", image);
        if (cv::waitKey(33) >= 0) {
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}
