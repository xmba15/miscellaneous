cmake_minimum_required(VERSION 3.10)

project(cublas_example LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)

cuda_add_executable(cublas_example
  ${PROJECT_SOURCE_DIR}/src/Main.cpp
  ${PROJECT_SOURCE_DIR}/src/CudaUtils.cu
  ${PROJECT_SOURCE_DIR}/src/SgemmGPU.cu
)

target_link_libraries(cublas_example
  ${CUDA_CUBLAS_LIBRARIES}
)
