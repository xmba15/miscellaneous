cmake_minimum_required(VERSION 2.6)

project(garborfilters CXX)
set(CMAKE_BUILD_TYPE RELEASE)

# OpenCV related
find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_executable(${CMAKE_PROJECT_NAME}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/garbor_filter.cpp
)

target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
