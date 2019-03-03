cmake_minimum_required(VERSION 2.6)

project(test_open3d)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE)
add_definitions("-Wall")

find_package(Open3D REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB BASIC_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.c
  )

add_executable(${CMAKE_PROJECT_NAME}
  ${BASIC_SOURCE_FILES}
  )
target_link_libraries(${CMAKE_PROJECT_NAME} ${Open3D_LIBRARIES})
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${Open3D_INCLUDE_DIRS})
