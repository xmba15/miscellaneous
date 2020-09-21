cmake_minimum_required(VERSION 3.0)

project(gmm)
set (CMAKE_CXX_STANDARD 17)

add_subdirectory(externals/matplotplusplus)

add_executable(${PROJECT_NAME}_1d_app
  ${PROJECT_SOURCE_DIR}/src/GMM1DApp.cpp
)

target_link_libraries(${PROJECT_NAME}_1d_app
  PUBLIC
    matplot
)
