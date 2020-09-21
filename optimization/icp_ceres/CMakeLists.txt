cmake_minimum_required(VERSION 3.10)

project(icp_ceres)

find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(PCL REQUIRED)

add_executable(${PROJECT_NAME}_app
  ${CMAKE_CURRENT_SOURCE_DIR}/src/App.cpp
)

target_link_libraries(${PROJECT_NAME}_app
  PUBLIC
    ${CERES_LIBRARIES}
    Sophus::Sophus
    ${PCL_LIBRARIES}
)

target_include_directories(${PROJECT_NAME}_app
  PUBLIC
    ${CERES_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)
