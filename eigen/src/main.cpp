// Copyright (c) 2018
// All Rights Reserved.
// Author: btran@btranPC (btran)

#include <iostream>
#include "TestEigen.hpp"

int main(int argc, char *argv[]) {
  Eigen::Matrix<float, 5, 6> m;
  m << 1, 0, 1, 0, 0, 0,
       0, 1, 0, 0, 0, 0,
       1, 1, 0, 0, 0, 0,
       1, 0, 0, 1, 1, 0,
       0, 0, 0, 1, 0, 1;

  std::cout << m << "\n";

  Eigen::JacobiSVD< Eigen::Matrix<float, 5, 6> > svd(m, Eigen::ComputeFullU |
                                        Eigen::ComputeFullV);

  std::cout << "singular values" << "\n"
            << svd.singularValues() << "\n\n";
  std::cout << "matrix U" << "\n" << svd.matrixU() << "\n\n";
  std::cout << "matrix V" << "\n" << svd.matrixV() << "\n\n";

  return 0;
}
