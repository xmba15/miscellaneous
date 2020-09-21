cmake_minimum_required(VERSION 3.10)

project(test_ceres)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE DEBUG)

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

add_subdirectory(examples)
