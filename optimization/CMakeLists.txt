cmake_minimum_required(VERSION 3.8)

project(project_name)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE DEBUG)
add_definitions("-Wall")

# find_package()
# include_directories()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(osqp)
__fetch_osqp(
  "${PROJECT_SOURCE_DIR}/cmake"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}_osqp"
)

include(osqp_eigen)
__fetch_osqp_eigen(
  "${PROJECT_SOURCE_DIR}/cmake"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}_osqp-eigen"
)

file(GLOB BASIC_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_executable(${PROJECT_NAME}
  ${BASIC_SOURCE_FILES}
)

target_link_libraries(${PROJECT_NAME} PRIVATE osqp)
