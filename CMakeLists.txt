cmake_minimum_required(VERSION 3.0)

project(miscellaneous)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE DEBUG)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(utility)

add_subdirectory(c_c++_code)
add_subdirectory(eigen)
add_subdirectory(open3d)
add_subdirectory(test_boost)
