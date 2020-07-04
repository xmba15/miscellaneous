cmake_minimum_required(VERSION 3.10)

project(test_ceres)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE DEBUG)

find_package(Ceres REQUIRED)

add_subdirectory(examples)
