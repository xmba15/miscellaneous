// code from
// https://github.com/robotology/osqp-eigen/blob/master/example/src/MPCExample.cpp

#include "Utility.hpp"
#include <iostream>

namespace test_osqp {

int testOsqp(void) {
  // set the preview window
  int mpcWindow = 10;

  // allocate the dynamics matrices
  Eigen::Matrix<double, 12, 12> a;
  Eigen::Matrix<double, 12, 4> b;

  // allocate the constraints vector
  Eigen::Matrix<double, 12, 1> xMax;
  Eigen::Matrix<double, 12, 1> xMin;
  Eigen::Matrix<double, 4, 1> uMax;
  Eigen::Matrix<double, 4, 1> uMin;

  // allocate the weight matrices
  Eigen::DiagonalMatrix<double, 12> Q;
  Eigen::DiagonalMatrix<double, 4> R;

  // allocate the initial and the reference state space
  Eigen::Matrix<double, 12, 1> x0;
  Eigen::Matrix<double, 12, 1> xRef;

  // allocate QP problem matrices and vectores
  Eigen::SparseMatrix<double> hessian;
  Eigen::VectorXd gradient;
  Eigen::SparseMatrix<double> linearMatrix;
  Eigen::VectorXd lowerBound;
  Eigen::VectorXd upperBound;

  // set the initial and the desired states
  x0 << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  xRef << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  // set MPC problem quantities
  setDynamicsMatrices(a, b);
  setInequalityConstraints(xMax, xMin, uMax, uMin);
  setWeightMatrices(Q, R);

  // cast the MPC problem as QP problem
  castMPCToQPHessian(Q, R, mpcWindow, hessian);
  castMPCToQPGradient(Q, xRef, mpcWindow, gradient);
  castMPCToQPConstraintMatrix(a, b, mpcWindow, linearMatrix);
  castMPCToQPConstraintVectors(xMax, xMin, uMax, uMin, x0, mpcWindow,
                               lowerBound, upperBound);

  // instantiate the solver
  OsqpEigen::Solver solver;

  // settings
  // solver.settings()->setVerbosity(false);
  solver.settings()->setWarmStart(true);

  // set the initial data of the QP solver
  solver.data()->setNumberOfVariables(12 * (mpcWindow + 1) + 4 * mpcWindow);
  solver.data()->setNumberOfConstraints(2 * 12 * (mpcWindow + 1) +
                                        4 * mpcWindow);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  if (!solver.data()->setHessianMatrix(hessian))
    return 1;
  if (!solver.data()->setGradient(gradient))
    return 1;
  if (!solver.data()->setLinearConstraintsMatrix(linearMatrix))
    return 1;
  if (!solver.data()->setLowerBound(lowerBound))
    return 1;
  if (!solver.data()->setUpperBound(upperBound))
    return 1;

  // instantiate the solver
  if (!solver.initSolver())
    return 1;

  // controller input and QPSolution vector
  Eigen::Vector4d ctr;
  Eigen::VectorXd QPSolution;

  // number of iteration steps
  int numberOfSteps = 1;

  for (int i = 0; i < numberOfSteps; i++) {
    // solve the QP problem
    if (!solver.solve())
      return 1;

    // get the controller input
    QPSolution = solver.getSolution();
    ctr = QPSolution.block(12 * (mpcWindow + 1), 0, 4, 1);

    // // save data into file
    // auto x0Data = x0.data();

    // propagate the model
    x0 = a * x0 + b * ctr;

    // update the constraint bound
    updateConstraintVectors(x0, lowerBound, upperBound);
    if (!solver.updateBounds(lowerBound, upperBound))
      return 1;
  }

  std::cout << "---------------------------------------------------------\n";
  std::cout << "---------------------------------------------------------\n";
  std::cout << ctr << "\n";
  std::cout << QPSolution.size() << "\n";

#pragma GCC diagnostic pop
  return 0;
}

} // namespace test_osqp

int main(int argc, char *argv[]) {
  test_osqp::testOsqp();
  return 0;
}
