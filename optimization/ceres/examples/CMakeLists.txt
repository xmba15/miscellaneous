cmake_minimum_required(VERSION 3.10)

add_executable(hello_world HelloWorld.cpp)
target_include_directories(hello_world
  PUBLIC
    {CERES_INCLUDE_DIRS}
)
target_link_directories(hello_world
  PUBLIC
    {CERES_LIBRARIES}
)
