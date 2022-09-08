cmake_minimum_required(VERSION 3.10)

project(sr-sift)

find_package(OpenCV 4.2 REQUIRED)

file(GLOB BASIC_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(${PROJECT_NAME}
  ${BASIC_SOURCE_FILES}
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    ${OpenCV_LIBRARIES}
)

target_include_directories(${PROJECT_NAME}
  SYSTEM PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)
