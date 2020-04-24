cmake_minimum_required(VERSION 3.10)

project(hello_world)

file(GLOB BASIC_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/HelloWorld.cu
)

cuda_add_executable(${PROJECT_NAME}
  ${BASIC_SOURCE_FILES}
)
