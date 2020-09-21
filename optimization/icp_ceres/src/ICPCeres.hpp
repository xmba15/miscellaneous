/**
 * @file    ICPCeres.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_types.h>

#include <sophus/se3.hpp>

#include <ceres/ceres.h>

namespace
{
// 3 residuals, 6 size of parameters
template <typename PointCloudType> class ProjectionErrorResidual : public ceres::SizedCostFunction<3, 6>
{
 public:
    ProjectionErrorResidual(const PointCloudType& srcPoint, const PointCloudType& dstPoint)
        : m_src(srcPoint.getVector3fMap().template cast<double>())
        , m_dst(dstPoint.getVector3fMap().template cast<double>())
    {
    }

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
    {
        Sophus::SE3d T = Sophus::SE3d::exp(Eigen::Map<const Eigen::Matrix<double, 6, 1>>(parameters[0]));
        Eigen::Map<Eigen::Vector3d> residualsV(residuals);

        Eigen::Vector3d projected = T * m_src;
        residualsV = m_dst - projected;

        if (jacobians && jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> JSE3(jacobians[0]);
            JSE3.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
            JSE3.block<3, 3>(0, 3) = Sophus::SO3d::hat(projected);
        }

        return true;
    }

 private:
    Eigen::Vector3d m_src;
    Eigen::Vector3d m_dst;
};

// 1 residual, 6 size of parameters
template <typename PointCloudType> class ProjectionErrorResidualPointToPlane : public ceres::SizedCostFunction<1, 6>
{
 public:
    ProjectionErrorResidualPointToPlane(const PointCloudType& srcPoint, const PointCloudType& dstPoint,
                                        const Eigen::Vector3f& normal)
        : m_src(srcPoint.getVector3fMap().template cast<double>())
        , m_dst(dstPoint.getVector3fMap().template cast<double>())
        , m_normal(normal.cast<double>())
    {
    }

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
    {
        Sophus::SE3d T = Sophus::SE3d::exp(Eigen::Map<const Eigen::Matrix<double, 6, 1>>(parameters[0]));

        Eigen::Vector3d projected = T * m_src;
        residuals[0] = m_normal.dot(m_dst - projected);

        if (jacobians && jacobians[0]) {
            Eigen::Matrix<double, 3, 6> A = Eigen::Matrix<double, 3, 6>::Zero();
            A.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
            A.block<3, 3>(0, 3) = Sophus::SO3d::hat(projected);

            Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor>> JSE3(jacobians[0]);
            JSE3.block<1, 6>(0, 0) = m_normal.transpose() * A;
        }

        return true;
    }

 private:
    Eigen::Vector3d m_src;
    Eigen::Vector3d m_dst;
    Eigen::Vector3d m_normal;
};
}  // namespace

namespace _pcl
{
template <typename PointCloudType>
void estimate3D3DPose(const typename pcl::PointCloud<PointCloudType>::Ptr& src,
                      const typename pcl::PointCloud<PointCloudType>::Ptr& dst, Eigen::Affine3f& pose)
{
    Sophus::SE3d poseSE3(pose.cast<double>().rotation(), pose.cast<double>().translation());
    // poseSE3.data() has length of 7; 4 for rotation in quaternion form and 3 for translation
    Eigen::Matrix<double, 6, 1> posese3 = poseSE3.log();

    ceres::Problem prob;

    // no point-to-point correspondence
    for (std::size_t i = 0; i < src->size(); ++i) {
        const auto& curSrcPoint = src->points[i];
        for (std::size_t j = 0; j < dst->size(); ++j) {
            const auto& curDstPoint = dst->points[i];
            ceres::CostFunction* costFunc = new ProjectionErrorResidual<PointCloudType>(curSrcPoint, curDstPoint);
            prob.AddResidualBlock(costFunc, nullptr, posese3.data());
        }
    }
    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &prob, &summary);

    std::cout << summary.BriefReport() << std::endl;

    poseSE3 = Sophus::SE3d::exp(posese3);
    pose.matrix() = poseSE3.matrix().cast<float>();
}

template <typename PointCloudType>
inline pcl::PointCloud<pcl::Normal> calculateNormals(const typename pcl::PointCloud<PointCloudType>::Ptr& inCloud,
                                                     const int numNeighbors = 3, const int numThreads = 4)
{
    pcl::PointCloud<pcl::Normal> normals;
    pcl::NormalEstimationOMP<PointCloudType, pcl::Normal> estimator;
    estimator.setNumberOfThreads(numThreads);
    estimator.setInputCloud(inCloud);

    typename pcl::search::KdTree<PointCloudType>::Ptr tree(new pcl::search::KdTree<PointCloudType>);
    estimator.setSearchMethod(tree);
    estimator.setKSearch(numNeighbors);
    estimator.compute(normals);

    return normals;
}

template <typename PointCloudType>
void estimate3D3DPosePointToPlane(const typename pcl::PointCloud<PointCloudType>::Ptr& src,
                                  const typename pcl::PointCloud<PointCloudType>::Ptr& dst, Eigen::Affine3f& pose)
{
    auto srcNormals = calculateNormals<PointCloudType>(src);
    auto dstNormals = calculateNormals<PointCloudType>(dst);

    Sophus::SE3d poseSE3(pose.cast<double>().rotation(), pose.cast<double>().translation());
    // poseSE3.data() has length of 7; 4 for rotation in quaternion form and 3 for translation
    Eigen::Matrix<double, 6, 1> posese3 = poseSE3.log();

    ceres::Problem prob;

    // no point-to-point correspondence
    for (std::size_t i = 0; i < src->size(); ++i) {
        const auto& curSrcPoint = src->points[i];
        const auto& curSrcNormal = srcNormals.points[i];
        if (curSrcNormal.getNormalVector3fMap().hasNaN()) {
            continue;
        }

        for (std::size_t j = 0; j < dst->size(); ++j) {
            const auto& curDstPoint = dst->points[i];
            const auto& curDstNormal = dstNormals.points[i];
            if (curDstNormal.getNormalVector3fMap().hasNaN()) {
                continue;
            }

            Eigen::Vector3f averageNormal = curSrcNormal.getNormalVector3fMap() + curDstNormal.getNormalVector3fMap();

            ceres::CostFunction* costFunc =
                new ProjectionErrorResidualPointToPlane<PointCloudType>(curSrcPoint, curDstPoint, averageNormal);
            prob.AddResidualBlock(costFunc, nullptr, posese3.data());
        }
    }
    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &prob, &summary);

    std::cout << summary.BriefReport() << std::endl;

    poseSE3 = Sophus::SE3d::exp(posese3);
    pose.matrix() = poseSE3.matrix().cast<float>();
}
}  // namespace _pcl
