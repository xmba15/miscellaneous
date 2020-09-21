cmake_minimum_required(VERSION 3.10)

project(opencv_dnn)
set (CMAKE_CXX_STANDARD 17)

set(OPENCV_DNN_MIN_VERSION 4.2)
find_package(OpenCV ${OPENCV_DNN_MIN_VERSION} REQUIRED)

add_executable(test_maskrcnn
  ${PROJECT_SOURCE_DIR}/src/MaskRCNN.cpp
)

target_include_directories(test_maskrcnn
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_maskrcnn
  PUBLIC
    ${OpenCV_LIBRARIES}
)

add_executable(yolov4
  ${PROJECT_SOURCE_DIR}/src/Yolov4.cpp
  ${PROJECT_SOURCE_DIR}/src/Yolov4App.cpp
)

target_include_directories(yolov4
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolov4
  PUBLIC
    ${OpenCV_LIBRARIES}
)
