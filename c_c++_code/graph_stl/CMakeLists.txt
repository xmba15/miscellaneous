cmake_minimum_required(VERSION 3.0)

project(graph)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE)
add_definitions("-Wall")

# find_package()
# include_directories()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB BASIC_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
  )

# DFS
add_executable(${PROJECT_NAME}_DFS
  ${CMAKE_CURRENT_SOURCE_DIR}/examples/Example1.cpp
  ${BASIC_SOURCE_FILES}
  )

# BFS
add_executable(${PROJECT_NAME}_BFS
  ${CMAKE_CURRENT_SOURCE_DIR}/examples/Example2.cpp
  ${BASIC_SOURCE_FILES}
  )

# Topological sort Kahn
add_executable(${PROJECT_NAME}_TopologicalSortKahn
  ${CMAKE_CURRENT_SOURCE_DIR}/examples/Example3.cpp
  ${BASIC_SOURCE_FILES}
  )
