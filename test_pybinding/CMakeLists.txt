cmake_minimum_required(VERSION 2.6)

project(example)
set(PYBIND11_CPP_STANDARD -std=c++11)

find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

file(GLOB BASIC_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  )

pybind11_add_module(${PROJECT_NAME} SHARED ${BASIC_SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::module ${OpenCV_LIBS})
