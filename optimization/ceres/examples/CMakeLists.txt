cmake_minimum_required(VERSION 3.10)

add_executable(example examples.cpp)
target_include_directories(example
  PUBLIC
    {CERES_INCLUDE_DIRS}
)
target_link_directories(example
  PUBLIC
    {CERES_LIBRARIES}
)
