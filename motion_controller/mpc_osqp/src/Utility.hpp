// code from
// https://github.com/robotology/osqp-eigen/blob/master/example/src/MPCExample.cpp

#include <Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>

namespace test_osqp {

void setDynamicsMatrices(Eigen::Matrix<double, 12, 12> &a,
                         Eigen::Matrix<double, 12, 4> &b) {
  a << 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
      0., 0.1, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0.,
      0.0488, 0., 0., 1., 0., 0., 0.0016, 0., 0., 0.0992, 0., 0., 0., -0.0488,
      0., 0., 1., 0., 0., -0.0016, 0., 0., 0.0992, 0., 0., 0., 0., 0., 0., 1.,
      0., 0., 0., 0., 0., 0.0992, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      0., 0., 0., 1., 0., 0., 0., 0.9734, 0., 0., 0., 0., 0., 0.0488, 0., 0.,
      0.9846, 0., 0., 0., -0.9734, 0., 0., 0., 0., 0., -0.0488, 0., 0., 0.9846,
      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.9846;

  b << 0., -0.0726, 0., 0.0726, -0.0726, 0., 0.0726, 0., -0.0152, 0.0152,
      -0.0152, 0.0152, -0., -0.0006, -0., 0.0006, 0.0006, 0., -0.0006, 0.0000,
      0.0106, 0.0106, 0.0106, 0.0106, 0, -1.4512, 0., 1.4512, -1.4512, 0.,
      1.4512, 0., -0.3049, 0.3049, -0.3049, 0.3049, -0., -0.0236, 0., 0.0236,
      0.0236, 0., -0.0236, 0., 0.2107, 0.2107, 0.2107, 0.2107;
}

void setInequalityConstraints(Eigen::Matrix<double, 12, 1> &xMax,
                              Eigen::Matrix<double, 12, 1> &xMin,
                              Eigen::Matrix<double, 4, 1> &uMax,
                              Eigen::Matrix<double, 4, 1> &uMin) {
  double u0 = 10.5916;

  // input inequality constraints
  uMin << 9.6 - u0, 9.6 - u0, 9.6 - u0, 9.6 - u0;

  uMax << 13 - u0, 13 - u0, 13 - u0, 13 - u0;

  // state inequality constraints
  xMin << -M_PI / 6, -M_PI / 6, -OsqpEigen::INFTY, -OsqpEigen::INFTY,
      -OsqpEigen::INFTY, -1., -OsqpEigen::INFTY, -OsqpEigen::INFTY,
      -OsqpEigen::INFTY, -OsqpEigen::INFTY, -OsqpEigen::INFTY,
      -OsqpEigen::INFTY;

  xMax << M_PI / 6, M_PI / 6, OsqpEigen::INFTY, OsqpEigen::INFTY,
      OsqpEigen::INFTY, OsqpEigen::INFTY, OsqpEigen::INFTY, OsqpEigen::INFTY,
      OsqpEigen::INFTY, OsqpEigen::INFTY, OsqpEigen::INFTY, OsqpEigen::INFTY;
}

void setWeightMatrices(Eigen::DiagonalMatrix<double, 12> &Q,
                       Eigen::DiagonalMatrix<double, 4> &R) {
  Q.diagonal() << 0, 0, 10., 10., 10., 10., 0, 0, 0, 5., 5., 5.;
  R.diagonal() << 0.1, 0.1, 0.1, 0.1;
}

void castMPCToQPHessian(const Eigen::DiagonalMatrix<double, 12> &Q,
                        const Eigen::DiagonalMatrix<double, 4> &R,
                        int mpcWindow,
                        Eigen::SparseMatrix<double> &hessianMatrix) {

  hessianMatrix.resize(12 * (mpcWindow + 1) + 4 * mpcWindow,
                       12 * (mpcWindow + 1) + 4 * mpcWindow);

  // populate hessian matrix
  for (int i = 0; i < 12 * (mpcWindow + 1) + 4 * mpcWindow; i++) {
    if (i < 12 * (mpcWindow + 1)) {
      int posQ = i % 12;
      float value = Q.diagonal()[posQ];
      if (value != 0)
        hessianMatrix.insert(i, i) = value;
    } else {
      int posR = i % 4;
      float value = R.diagonal()[posR];
      if (value != 0)
        hessianMatrix.insert(i, i) = value;
    }
  }
}

void castMPCToQPGradient(const Eigen::DiagonalMatrix<double, 12> &Q,
                         const Eigen::Matrix<double, 12, 1> &xRef,
                         int mpcWindow, Eigen::VectorXd &gradient) {

  Eigen::Matrix<double, 12, 1> Qx_ref;
  Qx_ref = Q * (-xRef);

  // populate the gradient vector
  gradient = Eigen::VectorXd::Zero(12 * (mpcWindow + 1) + 4 * mpcWindow, 1);
  for (int i = 0; i < 12 * (mpcWindow + 1); i++) {
    int posQ = i % 12;
    float value = Qx_ref(posQ, 0);
    gradient(i, 0) = value;
  }
}

void castMPCToQPConstraintMatrix(
    const Eigen::Matrix<double, 12, 12> &dynamicMatrix,
    const Eigen::Matrix<double, 12, 4> &controlMatrix, int mpcWindow,
    Eigen::SparseMatrix<double> &constraintMatrix) {
  constraintMatrix.resize(12 * (mpcWindow + 1) + 12 * (mpcWindow + 1) +
                              4 * mpcWindow,
                          12 * (mpcWindow + 1) + 4 * mpcWindow);

  // populate linear constraint matrix
  for (int i = 0; i < 12 * (mpcWindow + 1); i++) {
    constraintMatrix.insert(i, i) = -1;
  }

  for (int i = 0; i < mpcWindow; i++)
    for (int j = 0; j < 12; j++)
      for (int k = 0; k < 12; k++) {
        float value = dynamicMatrix(j, k);
        if (value != 0) {
          constraintMatrix.insert(12 * (i + 1) + j, 12 * i + k) = value;
        }
      }

  for (int i = 0; i < mpcWindow; i++)
    for (int j = 0; j < 12; j++)
      for (int k = 0; k < 4; k++) {
        float value = controlMatrix(j, k);
        if (value != 0) {
          constraintMatrix.insert(12 * (i + 1) + j,
                                  4 * i + k + 12 * (mpcWindow + 1)) = value;
        }
      }

  for (int i = 0; i < 12 * (mpcWindow + 1) + 4 * mpcWindow; i++) {
    constraintMatrix.insert(i + (mpcWindow + 1) * 12, i) = 1;
  }
}

void castMPCToQPConstraintVectors(const Eigen::Matrix<double, 12, 1> &xMax,
                                  const Eigen::Matrix<double, 12, 1> &xMin,
                                  const Eigen::Matrix<double, 4, 1> &uMax,
                                  const Eigen::Matrix<double, 4, 1> &uMin,
                                  const Eigen::Matrix<double, 12, 1> &x0,
                                  int mpcWindow, Eigen::VectorXd &lowerBound,
                                  Eigen::VectorXd &upperBound) {
  // evaluate the lower and the upper inequality vectors
  Eigen::VectorXd lowerInequality =
      Eigen::MatrixXd::Zero(12 * (mpcWindow + 1) + 4 * mpcWindow, 1);
  Eigen::VectorXd upperInequality =
      Eigen::MatrixXd::Zero(12 * (mpcWindow + 1) + 4 * mpcWindow, 1);
  for (int i = 0; i < mpcWindow + 1; i++) {
    lowerInequality.block(12 * i, 0, 12, 1) = xMin;
    upperInequality.block(12 * i, 0, 12, 1) = xMax;
  }
  for (int i = 0; i < mpcWindow; i++) {
    lowerInequality.block(4 * i + 12 * (mpcWindow + 1), 0, 4, 1) = uMin;
    upperInequality.block(4 * i + 12 * (mpcWindow + 1), 0, 4, 1) = uMax;
  }

  // evaluate the lower and the upper equality vectors
  Eigen::VectorXd lowerEquality =
      Eigen::MatrixXd::Zero(12 * (mpcWindow + 1), 1);
  Eigen::VectorXd upperEquality;
  lowerEquality.block(0, 0, 12, 1) = -x0;
  upperEquality = lowerEquality;
  lowerEquality = lowerEquality;

  // merge inequality and equality vectors
  lowerBound =
      Eigen::MatrixXd::Zero(2 * 12 * (mpcWindow + 1) + 4 * mpcWindow, 1);
  lowerBound << lowerEquality, lowerInequality;

  upperBound =
      Eigen::MatrixXd::Zero(2 * 12 * (mpcWindow + 1) + 4 * mpcWindow, 1);
  upperBound << upperEquality, upperInequality;
}

void updateConstraintVectors(const Eigen::Matrix<double, 12, 1> &x0,
                             Eigen::VectorXd &lowerBound,
                             Eigen::VectorXd &upperBound) {
  lowerBound.block(0, 0, 12, 1) = -x0;
  upperBound.block(0, 0, 12, 1) = -x0;
}

double getErrorNorm(const Eigen::Matrix<double, 12, 1> &x,
                    const Eigen::Matrix<double, 12, 1> &xRef) {
  // evaluate the error
  Eigen::Matrix<double, 12, 1> error = x - xRef;

  // return the norm
  return error.norm();
}

/**
 * @brief Overloading the << operator to quickly print out the content of the
 * containers.
 */
template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
std::ostream &operator<<(std::ostream &os, const Container<T> &container) {
  using ContainerType = Container<T>;
  for (typename ContainerType::const_iterator it = container.begin();
       it != container.end(); ++it) {
    os << *it << " ";
  }

  return os;
}

} // namespace test_osqp
