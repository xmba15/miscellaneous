cmake_minimum_required(VERSION 3.10)

project(knn_cuda LANGUAGES CXX)
set(LIBRARY_NAME ${PROJECT_NAME})

set (CMAKE_CXX_STANDARD 17)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else(CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA not found")
  return()
endif()

add_library(${LIBRARY_NAME}
  SHARED
    ${PROJECT_SOURCE_DIR}/src/knn.cpp
)

target_include_directories(${LIBRARY_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

set_source_files_properties(
  ${PROJECT_SOURCE_DIR}/src/knn.cpp
    PROPERTIES LANGUAGE CUDA
)

target_compile_options(${LIBRARY_NAME}
  INTERFACE
     $<$<CONFIG:Debug>:-O0 -g -Wall -Werror>
     $<$<CONFIG:Release>:-O3>
  PRIVATE
     $<$<COMPILE_LANGUAGE:CUDA>:-expt-extended-lambda>
)

add_subdirectory(examples)
