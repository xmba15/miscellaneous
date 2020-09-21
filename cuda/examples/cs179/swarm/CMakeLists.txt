cmake_minimum_required(VERSION 3.0)

project(swarm LANGUAGES CUDA CXX)
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
  src/main.cpp
  src/swarm.cuh
)

# set_target_properties(${PROJECT_NAME} PROPERTIES
#   CUDA_SEPARABLE_COMPILATION ON
# )

# target_link_libraries(${PROJECT_NAME})
