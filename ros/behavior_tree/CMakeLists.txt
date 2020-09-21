cmake_minimum_required(VERSION 3.10)

project(test_bhtree)
set (CMAKE_CXX_STANDARD 17)

find_package(BehaviorTreeV3)

add_executable(${PROJECT_NAME}_app
  ${PROJECT_SOURCE_DIR}/src/App.cpp
)

target_link_libraries(${PROJECT_NAME}_app
  BT::behaviortree_cpp_v3
)
