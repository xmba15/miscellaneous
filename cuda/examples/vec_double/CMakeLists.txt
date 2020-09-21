cmake_minimum_required(VERSION 3.10)

project(vec_double)

file(GLOB BASIC_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/VecDouble.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

cuda_add_executable(${PROJECT_NAME}
  ${BASIC_SOURCE_FILES}
)
