/**
 * @file    OpticalFlow.cpp
 *
 * @author  btran
 *
 */

#include "OpticalFlow.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

namespace _cv
{
void OpticalFlowTracker::operator()(const cv::Range& range) const
{
    for (std::size_t i = range.start; i < range.end; ++i) {
        const auto& kp = m_kp1[i];
        double dx = 0, dy = 0;  // dx,dy need to be estimated
        if (m_hasInitialGuess) {
            dx = m_kp2[i].pt.x - kp.pt.x;
            dy = m_kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        m_success[i] = true;

        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        Eigen::Vector2d b = Eigen::Vector2d::Zero();
        Eigen::Vector2d J;
        for (int iter = 0; iter < m_numIterations; iter++) {
            if (m_inverse == false) {
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            } else {
                // only reset b
                b = Eigen::Vector2d::Zero();
            }

            cost = 0;

            // compute cost and jacobian
            for (int x = -m_halfWindowSize; x <= m_halfWindowSize; x++)
                for (int y = -m_halfWindowSize; y <= m_halfWindowSize; y++) {
                    double error = getPixelValue(m_img1, kp.pt.x + x, kp.pt.y + y) -
                                   getPixelValue(m_img2, kp.pt.x + x + dx, kp.pt.y + y + dy);

                    if (m_inverse == false) {
                        J = -1.0 * Eigen::Vector2d(imageDerivativeX(m_img2, kp.pt.x + dx + x, kp.pt.y + dy + y),
                                                   imageDerivativeY(m_img2, kp.pt.x + dx + x, kp.pt.y + dy + y));
                    } else if (iter == 0) {
                        // in inverse mode, J keeps same for all m_numIterations
                        // this J does not change when dx, dy is updated, so we can store it and only compute error
                        J = -1.0 * Eigen::Vector2d(imageDerivativeX(m_img1, kp.pt.x + dx + x, kp.pt.y + dy + y),
                                                   imageDerivativeY(m_img1, kp.pt.x + dx + x, kp.pt.y + dy + y));
                    }
                    // compute H, b and set cost;
                    b += -error * J;
                    cost += error * error;
                    if (m_inverse == false || iter == 0) {
                        // also update H
                        H += J * J.transpose();
                    }
                }

            Eigen::Vector2d update = H.ldlt().solve(b);

            if (std::isnan(update[0])) {
                std::cerr << "update is nan" << std::endl;
                m_success[i] = false;
                break;
            }

            if (iter > 0 && cost > lastCost) {
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            m_success[i] = true;

            if (update.norm() < m_epsilon) {
                break;
            }
        }

        m_kp2[i].pt = kp.pt + cv::Point2f(dx, dy);
    }
}
}  // namespace _cv
