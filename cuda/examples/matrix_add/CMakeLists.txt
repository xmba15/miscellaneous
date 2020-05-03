cmake_minimum_required(VERSION 3.10)

project(matrix_add)

file(GLOB BASIC_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
)

cuda_add_executable(${PROJECT_NAME}
  ${BASIC_SOURCE_FILES}
)
