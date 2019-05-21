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

add_executable(${PROJECT_NAME}_Example1
  ${CMAKE_CURRENT_SOURCE_DIR}/examples/Example1.cpp
  ${BASIC_SOURCE_FILES}
  )

add_executable(${PROJECT_NAME}_Example2
  ${CMAKE_CURRENT_SOURCE_DIR}/examples/Example2.cpp
  ${BASIC_SOURCE_FILES}
  )
