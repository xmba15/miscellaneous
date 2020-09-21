/**
 * @file    DirectPoseEstimation.cpp
 *
 * @author  btran
 *
 */

#include <ceres/ceres.h>

#include "DirectPoseEstimation.hpp"

#include "Utility.hpp"

namespace _cv
{
namespace
{
template <int HalfWindowSize = 1>
class DirectPoseEstimationResidual
    : public ceres::SizedCostFunction<(2 * HalfWindowSize + 1) * (2 * HalfWindowSize + 1), 6>
{
 public:
    DirectPoseEstimationResidual(const cv::Mat& img1, const cv::Mat& img2, const cv::Point2d& refPoint, double depth,
                                 const _cv::CameraMatrix& K, cv::Point2d& projected)
        : m_img1(img1)
        , m_img2(img2)
        , m_refPoint(refPoint)
        , m_depth(depth)
        , m_K(K)
        , m_projected(projected)
    {
        m_refPoint3d = m_depth * Eigen::Vector3d((m_refPoint.x - m_K.cx) / m_K.fx, (m_refPoint.y - m_K.cy) / m_K.fy, 1);
    }

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
    {
        Sophus::SE3d T = Sophus::SE3d::exp(Eigen::Map<const Eigen::Matrix<double, 6, 1>>(parameters[0]));
        std::fill(residuals, residuals + (2 * HalfWindowSize + 1) * (2 * HalfWindowSize + 1), 0);
        if (jacobians && jacobians[0]) {
            std::fill(jacobians[0], jacobians[0] + (2 * HalfWindowSize + 1) * (2 * HalfWindowSize + 1) * 6, 0);
        }

        Eigen::Vector3d curPoint3d = T * m_refPoint3d;
        if (curPoint3d[2] < 0) {
            return true;
        }

        double u = m_K.fx * curPoint3d[0] / curPoint3d[2] + m_K.cx, v = m_K.fy * curPoint3d[1] / curPoint3d[2] + m_K.cy;
        if (u < HalfWindowSize || u > m_img2.cols - HalfWindowSize || v < HalfWindowSize ||
            v > m_img2.rows - HalfWindowSize) {
            return true;
        }

        m_projected.x = u;
        m_projected.y = v;

        double X = curPoint3d[0], Y = curPoint3d[1], Z = curPoint3d[2], Z2 = Z * Z, Z_inv = 1.0 / Z,
               Z2_inv = Z_inv * Z_inv;

        Eigen::Matrix<double, 2, 6> JPixelXi;
        JPixelXi(0, 0) = m_K.fx * Z_inv;
        JPixelXi(0, 1) = 0;
        JPixelXi(0, 2) = -m_K.fx * X * Z2_inv;
        JPixelXi(0, 3) = -m_K.fx * X * Y * Z2_inv;
        JPixelXi(0, 4) = m_K.fx + m_K.fx * X * X * Z2_inv;
        JPixelXi(0, 5) = -m_K.fx * Y * Z_inv;

        JPixelXi(1, 0) = 0;
        JPixelXi(1, 1) = m_K.fy * Z_inv;
        JPixelXi(1, 2) = -m_K.fy * Y * Z2_inv;
        JPixelXi(1, 3) = -m_K.fy - m_K.fy * Y * Y * Z2_inv;
        JPixelXi(1, 4) = m_K.fy * X * Y * Z2_inv;
        JPixelXi(1, 5) = m_K.fy * X * Z_inv;

        for (int y = -HalfWindowSize; y <= HalfWindowSize; ++y) {
            for (int x = -HalfWindowSize; x <= HalfWindowSize; ++x) {
                int curIdx = (y + HalfWindowSize) * (2 * HalfWindowSize + 1) + (x + HalfWindowSize);
                residuals[curIdx] =
                    getPixelValue(m_img1, m_refPoint.x + x, m_refPoint.y + y) - getPixelValue(m_img2, u + x, v + y);
                Eigen::Vector2d JImgPixel =
                    Eigen::Vector2d(imageDerivativeX(m_img2, u + x, v + y), imageDerivativeY(m_img2, u + x, v + y));

                if (jacobians && jacobians[0]) {
                    Eigen::Map<
                        Eigen::Matrix<double, (2 * HalfWindowSize + 1) * (2 * HalfWindowSize + 1), 6, Eigen::RowMajor>>
                        JSE3(jacobians[0]);
                    JSE3.row(curIdx) = -1.0 * (JImgPixel.transpose() * JPixelXi);
                }
            }
        }

        return true;
    }

 private:
    const cv::Mat& m_img1;
    const cv::Mat& m_img2;
    const cv::Point2d& m_refPoint;
    Eigen::Vector3d m_refPoint3d;
    double m_depth;
    const _cv::CameraMatrix& m_K;
    cv::Point2d& m_projected;
};
}  // namespace

void calcDirectPoseEstimationSingleLayer(const cv::Mat& img1, const cv::Mat& img2,
                                         const std::vector<cv::Point2d>& refPoints, const std::vector<double>& depth,
                                         const _cv::CameraMatrix& K, Sophus::SE3d& T21,
                                         std::vector<cv::Point2d>& projecteds, int numIterations)
{
    Eigen::Matrix<double, 6, 1> posese3 = T21.log();
    ceres::Problem prob;

    projecteds.resize(refPoints.size());
    std::fill(projecteds.begin(), projecteds.end(), cv::Point2d(0, 0));

    for (std::size_t i = 0; i < refPoints.size(); ++i) {
        const auto& curRefPoint = refPoints[i];
        ceres::CostFunction* costFunc =
            new DirectPoseEstimationResidual<1>(img1, img2, refPoints[i], depth[i], K, projecteds[i]);
        prob.AddResidualBlock(costFunc, nullptr, posese3.data());
    }

    ceres::Solver::Options options;
    options.max_num_iterations = numIterations;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &prob, &summary);

    T21 = Sophus::SE3d::exp(posese3);
}

void calcDirectPoseEstimationMultiLayer(const cv::Mat& img1, const cv::Mat& img2,
                                        const std::vector<cv::Point2d>& refPoints, const std::vector<double>& depth,
                                        const _cv::CameraMatrix& K, Sophus::SE3d& T21,
                                        std::vector<cv::Point2d>& projecteds, int numScale, double scaleFactor,
                                        int numIterations)
{
    std::vector<double> scales = {1.};
    for (int i = 1; i < numScale; ++i) {
        scales.emplace_back(scales.back() * scaleFactor);
    }

    auto pyr1 = createImagePyramid(img1, numScale, scaleFactor);
    auto pyr2 = createImagePyramid(img2, numScale, scaleFactor);

    for (int i = numScale - 1; i >= 0; --i) {
        auto curK = K.scale(scales[i]);
        std::vector<cv::Point2d> curPoints;
        curPoints.reserve(refPoints.size());
        std::transform(refPoints.begin(), refPoints.end(), std::back_inserter(curPoints),
                       [&scales, i](const auto& elem) { return elem * scales[i]; });
        calcDirectPoseEstimationSingleLayer(pyr1[i], pyr2[i], curPoints, depth, curK, T21, projecteds, numIterations);
    }
}
}  // namespace _cv
