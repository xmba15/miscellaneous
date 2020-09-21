cmake_minimum_required(VERSION 3.10)

project(keypoints_detection)
set (CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME}_harris_corner
  ${PROJECT_SOURCE_DIR}/src/HarrisCornerApp.cpp
)

target_include_directories(${PROJECT_NAME}_harris_corner
  SYSTEM PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}_harris_corner
  PUBLIC
    ${OpenCV_LIBRARIES}
)

# ------------------------------------------------------------------------------

add_executable(${PROJECT_NAME}_feature_matching
  ${PROJECT_SOURCE_DIR}/src/FeatureMatchingApp.cpp
)

target_include_directories(${PROJECT_NAME}_feature_matching
  SYSTEM PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}_feature_matching
  PUBLIC
    ${OpenCV_LIBRARIES}
)
