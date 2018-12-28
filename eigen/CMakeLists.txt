cmake_minimum_required(VERSION 2.6)

project(test_eigen)
set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
  )

add_executable(${CMAKE_PROJECT_NAME}
  ${FILES}
  )

target_link_libraries(${CMAKE_PROJECT_NAME})
