cmake_minimum_required(VERSION 3.0)

project(test_osqp)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE DEBUG)
add_definitions("-Wall")

file(GLOB BASIC_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
  )

find_package(OsqpEigen REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(${PROJECT_NAME}
  ${BASIC_SOURCE_FILES}
)

target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
  OsqpEigen::OsqpEigen
  osqp::osqp
)
