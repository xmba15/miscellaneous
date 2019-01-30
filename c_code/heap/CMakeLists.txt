cmake_minimum_required(VERSION 2.6)

project(heap)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE)
add_definitions("-Wall")

find_package(Boost COMPONENTS filesystem graph)
# include_directories()

file(GLOB FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.c
  )

add_executable(${CMAKE_PROJECT_NAME}
  ${FILES}
  )

target_link_libraries(${CMAKE_PROJECT_NAME} Boost::filesystem Boost::graph)
