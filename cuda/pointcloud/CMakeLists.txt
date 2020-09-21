cmake_minimum_required(VERSION 3.10)

project(simple_gpu_pcl_procssessing LANGUAGES CXX)
set(LIBRARY_NAME ${PROJECT_NAME})

set (CMAKE_CXX_STANDARD 17)

find_package(PCL REQUIRED)
find_package(CUDA REQUIRED)

# list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")

cuda_add_executable(down_sample_app
  ${PROJECT_SOURCE_DIR}/src/DownSampleApp.cpp
  ${PROJECT_SOURCE_DIR}/src/CudaUtils.cu
  ${PROJECT_SOURCE_DIR}/src/DownSampleHandler.cu
)

target_link_libraries(down_sample_app
  ${PCL_LIBRARIES}
)
target_include_directories(down_sample_app
  PUBLIC
    ${PCL_INCLUDE_DIRS}
)

# target_compile_options(down_sample_app
#   PRIVATE
#      $<$<COMPILE_LANGUAGE:CUDA>:-expt-extended-lambda>
# )
