cmake_minimum_required(VERSION 2.6)

project(test_qt5)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE)
add_definitions("-Wall")

find_package(Qt5 COMPONENTS Core Widgets  REQUIRED)
# message(STATUS "${Qt5Core_INCLUDE_DIRS}")

file(GLOB FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.c
)

add_executable(${CMAKE_PROJECT_NAME}
  ${FILES}
)

QT5_USE_MODULES(${CMAKE_PROJECT_NAME} Widgets Core)

# target_link_libraries(${CMAKE_PROJECT_NAME})
