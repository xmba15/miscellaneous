cmake_minimum_required(VERSION 3.10)

project(lab1 LANGUAGES CUDA CXX)
set (CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(FindSndFile)

if(SNDFILE_FOUND)
  add_compile_options(-DAUDIO_ON)
else()
  message(STATUS "sndfile not found")
  return()
endif()

include(CheckLanguage)
check_language(CUDA)

if(NOT CMAKE_CUDA_COMPILER)
  message(STATUS "cuda not found")
else()
  enable_language(CUDA)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_executable(${PROJECT_NAME}
  src/blur.cpp
  src/blur.cu
  src/blur.cuh
  src/cuda_header.cuh
)

target_include_directories(${PROJECT_NAME}
  SYSTEM PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${SNDFILE_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    ${SNDFILE_LIBRARIES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
