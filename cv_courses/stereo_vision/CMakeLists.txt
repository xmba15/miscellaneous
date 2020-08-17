cmake_minimum_required(VERSION 3.10)

project(stereo_vision)
set (CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)

list(APPEND COMMON_FILES
  ${PROJECT_SOURCE_DIR}/src/StereoEngine.cpp
)

add_executable(${PROJECT_NAME}_bm
  ${PROJECT_SOURCE_DIR}/src/BlockMatchingApp.cpp
  ${PROJECT_SOURCE_DIR}/src/BlockMatching.cpp
  ${COMMON_FILES}
)
target_link_libraries(${PROJECT_NAME}_bm
  PUBLIC
    ${OpenCV_LIBRARIES}
    stdc++fs
)
target_include_directories(${PROJECT_NAME}_bm
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)
