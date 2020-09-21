cmake_minimum_required(VERSION 3.8)

project(threading)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Threads REQUIRED)
add_subdirectory(examples)
