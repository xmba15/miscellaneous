cmake_minimum_required(VERSION 3.10)

project(shared_memory)

file(GLOB BASIC_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SharedMemory.cu
)

cuda_add_executable(${PROJECT_NAME}
  ${BASIC_SOURCE_FILES}
)
