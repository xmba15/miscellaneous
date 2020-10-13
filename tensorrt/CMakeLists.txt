cmake_minimum_required(VERSION 3.10)

project(testtensorrt)
set (CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(testtensort
  ${PROJECT_SOURCE_DIR}/main.cpp
  ${PROJECT_SOURCE_DIR}/TensorRTHandler.cpp
)

target_include_directories(testtensort
  SYSTEM PUBLIC
    ${OpenCV_INCLUDE_DIRS}
  PRIVATE
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(testtensort
  PUBLIC
    TensorRT::TensorRT
    nvonnxparser
    ${OpenCV_LIBRARIES}
  PRIVATE
  ${CUDA_LIBRARIES}
)
