cmake_minimum_required(VERSION 3.8)

project(osqp_eigen-download NONE)

include(ExternalProject)
ExternalProject_Add(
  osqp_eigen
  SOURCE_DIR "@OSQP_EIGEN_DOWNLOAD_ROOT@/osqp_eigen-src"
  BINARY_DIR "@OSQP_EIGEN_DOWNLOAD_ROOT@/osqp_eigen-build"
  GIT_REPOSITORY
    https://github.com/robotology/osqp-eigen
  GIT_TAG
    v0.4.1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
