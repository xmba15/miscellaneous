cmake_minimum_required(VERSION 3.10)

find_package(OpenCV REQUIRED)

add_executable(knn_app
  ${CMAKE_CURRENT_LIST_DIR}/App.cpp
)

target_link_libraries(knn_app
  PUBLIC
    ${LIBRARY_NAME}
    ${OpenCV_LIBRARIES}
)

target_include_directories(knn_app
  SYSTEM PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)
