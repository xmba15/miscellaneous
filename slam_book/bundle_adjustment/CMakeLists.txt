cmake_minimum_required(VERSION 3.10)

project(bundle_adjustment)

find_package(Ceres REQUIRED)
find_package(g2o REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME}_app
  ${CMAKE_CURRENT_SOURCE_DIR}/src/App.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/BALParser.cpp
)

target_link_libraries(${PROJECT_NAME}_app
  PUBLIC
    ${CERES_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

target_include_directories(${PROJECT_NAME}_app
  PUBLIC
    ${CERES_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# g2o::core
