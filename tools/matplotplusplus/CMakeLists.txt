cmake_minimum_required(VERSION 3.10)

project(testmatplotplusplus)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE DEBUG)

find_package(Matplot++ REQUIRED)

add_executable(${PROJECT_NAME}
  main.cpp
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    Matplot++::matplot
)
