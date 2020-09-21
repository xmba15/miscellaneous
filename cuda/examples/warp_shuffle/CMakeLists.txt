cmake_minimum_required(VERSION 3.10)

project(warp_shuffle LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)

cuda_add_executable(wapr_shuffle
  ${PROJECT_SOURCE_DIR}/src/Main.cu
)
