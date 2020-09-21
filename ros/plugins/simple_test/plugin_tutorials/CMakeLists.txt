cmake_minimum_required(VERSION 3.10)
project(plugin_tutorials)

find_package(catkin REQUIRED COMPONENTS
  pluginlib
  roscpp
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES plugin_tutorials
  CATKIN_DEPENDS pluginlib roscpp
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(polygon_plugins
  src/polygon_plugins.cpp
)

add_executable(polygon_loader
  src/polygon_loader.cpp
)

target_link_libraries(polygon_loader
  ${catkin_LIBRARIES}
  polygon_plugins
)
