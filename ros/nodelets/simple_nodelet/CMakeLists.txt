cmake_minimum_required(VERSION 3.10)
project(simple_nodelet)

find_package(catkin REQUIRED COMPONENTS
  nodelet
  pluginlib
  roscpp
)

catkin_package(
  LIBRARIES simple_nodelet
  CATKIN_DEPENDS nodelet pluginlib roscpp
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/plugin_nodelet_talker.cpp
  src/plugin_nodelet_listener.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
)
