cmake_minimum_required(VERSION 3.10)

project(lab2 LANGUAGES CUDA CXX)
set (CMAKE_CXX_STANDARD 17)

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
  src/ta_utilities.cpp
  src/ta_utilities.hpp
  src/transpose_device.cu
  src/transpose_device.cuh
  src/transpose_host.cpp
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
