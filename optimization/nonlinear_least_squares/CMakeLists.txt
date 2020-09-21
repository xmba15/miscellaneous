cmake_minimum_required(VERSION 3.10)

project(nls)
set (CMAKE_CXX_STANDARD 17)

add_subdirectory(externals/matplotplusplus)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(g2o REQUIRED)

add_executable(${PROJECT_NAME}_Gauss_Newton
  ${CMAKE_CURRENT_SOURCE_DIR}/src/GaussianNewtonExample.cpp
)

target_link_libraries(${PROJECT_NAME}_Gauss_Newton
  PUBLIC
    matplot
    Eigen3::Eigen
)

add_executable(${PROJECT_NAME}_Ceres
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CeresExample.cpp
)

target_link_libraries(${PROJECT_NAME}_Ceres
  PUBLIC
    matplot
    Eigen3::Eigen
    ${CERES_LIBRARIES}
)

target_link_directories(${PROJECT_NAME}_Ceres
  PUBLIC
    ${CERES_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_g2o
  ${CMAKE_CURRENT_SOURCE_DIR}/src/g2oExample.cpp
)

target_link_libraries(${PROJECT_NAME}_g2o
  PUBLIC
    matplot
    Eigen3::Eigen
    g2o::core
)
