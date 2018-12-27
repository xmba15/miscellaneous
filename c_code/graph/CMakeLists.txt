cmake_minimum_required(VERSION 2.6)

project(test_graph)
set(CMAKE_BUILD_TYPE)
set(CMAKE_CXX_STANDARD 11)

# find_package()
# include_directories()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB CFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/graph.c
  )

file(GLOB CPPFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/graph.cpp
  )

add_executable(${CMAKE_PROJECT_NAME}_c
  ${CFILES}
  )

add_executable(${CMAKE_PROJECT_NAME}_cpp
  ${CPPFILES}
  )
