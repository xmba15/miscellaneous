/**
 * @file    test.cpp
 *
 * @author  btran
 *
 * Copyright (c) organization
 *
 */
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <deque>
#include <iostream>
#include <numeric>
#include <random>

int main(int argc, char *argv[]) {
  double lower_bound = 0;
  double upper_bound = 10;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  std::default_random_engine re;

  std::vector<Eigen::MatrixXd> As;
  std::vector<Eigen::MatrixXd> Bs;

  int L = 10;
  int nz = 4;
  int nu = 2;

  As.reserve(L);
  Bs.reserve(L);

  for (int i = 0; i < L; ++i) {
    Eigen::MatrixXd tmp(nz, nz);
    tmp << unif(re), unif(re), unif(re), unif(re), unif(re), unif(re), unif(re),
        unif(re), unif(re), unif(re), unif(re), unif(re), unif(re), unif(re),
        unif(re), unif(re);

    As.emplace_back(tmp);

    Eigen::MatrixXd tmp2(nz, nu);
    tmp2 << unif(re), unif(re), unif(re), unif(re), unif(re), unif(re),
        unif(re), unif(re);
    Bs.emplace_back(tmp2);
  }

  std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>
      accumAs;
  Eigen::MatrixXd initAs = Eigen::MatrixXd::Ones(nz, nz);
  for (auto it = As.cbegin(); it != As.cend(); ++it) {
    initAs = (*it) * initAs;
    accumAs.emplace_back(initAs);
  }

  Eigen::MatrixXd GMatrix(nz * L, nz * L);

  std::vector<Eigen::MatrixXd> GMatrixRowVec;
  Eigen::MatrixXd GZeroMatrix = Eigen::MatrixXd::Zero(nz, nz);
  Eigen::MatrixXd GOneMatrix = Eigen::MatrixXd::Ones(nz, nz);

  for (int i = 0; i < L; ++i) {
    Eigen::MatrixXd GMatrixRow(nz, nz * L);
    int idx = 0;

    for (auto it = GMatrixRowVec.begin(); it != GMatrixRowVec.end(); ++it) {
      GMatrixRow.block(0, idx, nz, nz) = *it;
      *it = As[i] * (*it);
      idx += nz;
    }
    GMatrixRow.block(0, idx, nz, nz) = GOneMatrix;
    idx += nz;

    while (idx < nz * L) {
      GMatrixRow.block(0, idx, nz, nz) = GZeroMatrix;
      idx += nz;
    }

    GMatrix.block(i * nz, 0, nz, nz * L) = GMatrixRow;
    GMatrixRowVec.emplace_back(As[i]);
  }

  Eigen::MatrixXd AllBMatrix(nz * L, nu * L);

  for (int i = 0; i < L; ++i) {
    AllBMatrix.block(i * nz, i * nu, nz, nu) = Bs[i];
  }

  std::cout << GMatrix * AllBMatrix << "\n";

  std::cout << "---------------------------------------------------------------"
               "---------------------------------------------------------------"
            << "\n";
  std::cout << "---------------------------------------------------------------"
               "---------------------------------------------------------------"
            << "\n";

  Eigen::MatrixXd AnoGMatrix(nz * L, nu * L);

  std::vector<Eigen::MatrixXd> AnoGMatrixRowVec;
  Eigen::MatrixXd AnoGZeroMatrix = Eigen::MatrixXd::Zero(nz, nu);
  Eigen::MatrixXd AnoGOneMatrix = Eigen::MatrixXd::Ones(nz, nu);

  for (int i = 0; i < L; ++i) {
    Eigen::MatrixXd AnoGMatrixRow(nz, nu * L);
    int idx = 0;

    AnoGMatrixRowVec.emplace_back(Bs[0]);
    for (auto it = AnoGMatrixRowVec.begin(); it != AnoGMatrixRowVec.end();
         ++it) {
      AnoGMatrixRow.block(0, idx, nz, nu) = *it;
      *it = As[i] * (*it);
      idx += nu;
    }

    while (idx < nu * L) {
      AnoGMatrixRow.block(0, idx, nz, nu) = AnoGZeroMatrix;
      idx += nz;
    }

    AnoGMatrix.block(i * nz, 0, nz, nu * L) = AnoGMatrixRow;
    // AnoGMatrixRowVec.emplace_back(As[i]);
  }

  std::cout << AnoGMatrix << "\n";

  return 0;
}
