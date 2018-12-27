cmake_minimum_required(VERSION 2.6)

project(test_graph)
set(CMAKE_BUILD_TYPE)

# find_package()
# include_directories()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/graph.c
  )

add_executable(${CMAKE_PROJECT_NAME}
  ${FILES}
  )

target_link_libraries(${CMAKE_PROJECT_NAME})
