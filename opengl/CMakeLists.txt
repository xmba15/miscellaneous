cmake_minimum_required(VERSION 3.0)

project(test_opengl)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

add_subdirectory(examples)
