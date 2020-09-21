cmake_minimum_required(VERSION 3.8)

project(osqp-download NONE)

include(ExternalProject)
ExternalProject_Add(
  osqp
  SOURCE_DIR "@OSQP_DOWNLOAD_ROOT@/osqp-src"
  BINARY_DIR "@OSQP_DOWNLOAD_ROOT@/osqp-build"
  GIT_REPOSITORY
    https://github.com/oxfordcontrol/osqp
  GIT_TAG
    v0.6.0
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
