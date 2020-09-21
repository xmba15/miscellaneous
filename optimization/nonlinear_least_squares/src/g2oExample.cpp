/**
 * @file    g2oExample.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include <g2o/core/auto_differentiation.h>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <Eigen/Core>

#include "AppUtility.hpp"

class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d*>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl() override
    {
        *_estimate << 0, 0, 0;
    }

    virtual void oplusImpl(const double* update) override
    {
        *_estimate += Eigen::Vector3d(update);
    }

    virtual bool read(std::istream& in) override
    {
        return false;
    }

    virtual bool write(std::ostream& out) const override
    {
        return false;
    }
};

class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static CurveFittingEdge* getInstance(double x)
    {
        CurveFittingEdge* e = new CurveFittingEdge(x);
        return e;
    }

    virtual void computeError() override
    {
        const CurveFittingVertex* v = reinterpret_cast<const CurveFittingVertex*>(_vertices[0]);
        const auto& coeffs = *v->estimate();
        _error(0, 0) = _measurement - std::exp(coeffs[0] * _x * _x + coeffs[1] * _x + coeffs[2]);
    }

    virtual void linearizeOplus() override
    {
        const CurveFittingVertex* v = reinterpret_cast<const CurveFittingVertex*>(_vertices[0]);
        const auto& coeffs = *v->estimate();
        double y = std::exp(coeffs[0] * _x * _x + coeffs[1] * _x + coeffs[2]);
        _jacobianOplusXi[0] = -_x * _x * y;
        _jacobianOplusXi[1] = -_x * y;
        _jacobianOplusXi[2] = -y;
    }

    virtual bool read(std::istream& in) override
    {
        return false;
    }

    virtual bool write(std::ostream& out) const override
    {
        return false;
    }

 private:
    CurveFittingEdge() = delete;

    CurveFittingEdge(double x)
        : g2o::BaseUnaryEdge<1, double, CurveFittingVertex>()
        , _x(x)
    {
    }

 private:
    double _x;
};

int main(int argc, char* argv[])
{
    std::vector<double> sampleXs, sampleYs;
    ::generateSamples(NUM_SAMPLES, sampleXs, sampleYs);

    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())));
    optimizer.setVerbose(true);
    CurveFittingVertex v;
    Eigen::Vector3d coeffs(2, -1, 5);
    v.setEstimate(&coeffs);
    v.setId(0);
    optimizer.addVertex(new CurveFittingVertex(std::move(v)));

    for (int i = 0; i < NUM_SAMPLES; ++i) {
        CurveFittingEdge* e = CurveFittingEdge::getInstance(sampleXs[i]);
        e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        e->setId(i);
        e->setVertex(0, optimizer.vertex(0));
        e->setMeasurement(sampleYs[i]);
        optimizer.addEdge(e);
    }

    int numIteration = 100;
    optimizer.initializeOptimization();
    optimizer.optimize(numIteration);
    std::cout << coeffs << "\n";

    auto nonLinearFunc = [](const Eigen::Vector3d& coeffs, double x) {
        return std::exp(coeffs[0] * x * x + coeffs[1] * x + coeffs[2]);
    };
    std::vector<double> estimatedYs;
    estimatedYs.reserve(sampleXs.size());
    std::transform(sampleXs.begin(), sampleXs.end(), std::back_inserter(estimatedYs),
                   [&nonLinearFunc, &coeffs](double x) { return nonLinearFunc(coeffs, x); });
    ::visualize(sampleXs, sampleYs, estimatedYs);

    return EXIT_SUCCESS;
}
