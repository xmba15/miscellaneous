cmake_minimum_required(VERSION 2.6)

project(test_pybinding)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE)
add_definitions("-Wall")

find_package(pybind11 REQUIRED)

# include_directories()

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# file(GLOB BASIC_SOURCE_FILES
#   ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
#   ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
#   )

# add_executable(${CMAKE_PROJECT_NAME}
#   ${BASIC_SOURCE_FILES}
#   )

# target_link_libraries(${CMAKE_PROJECT_NAME})
