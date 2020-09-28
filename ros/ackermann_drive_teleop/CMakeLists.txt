cmake_minimum_required(VERSION 3.10)
project(ackermann_drive_teleop)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  ackermann_msgs
)

catkin_package(
 CATKIN_DEPENDS
   roscpp
   sensor_msgs
   ackermann_msgs
 DEPENDS
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(joystick_teleop
  src/joystick_teleop_node.cpp
)
target_link_libraries(joystick_teleop
  ${catkin_LIBRARIES}
)
