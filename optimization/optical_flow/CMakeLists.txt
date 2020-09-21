cmake_minimum_required(VERSION 3.10)

project(optical_flow)

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)

add_executable(${PROJECT_NAME}_app
  ${CMAKE_CURRENT_SOURCE_DIR}/src/App.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/OpticalFlow.cpp
)

target_link_libraries(${PROJECT_NAME}_app
  PUBLIC
    ${CERES_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

target_include_directories(${PROJECT_NAME}_app
  PUBLIC
    ${CERES_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(dense_optical_flow_app
  ${CMAKE_CURRENT_SOURCE_DIR}/src/DenseOpticalFlowApp.cpp
)

target_link_libraries(dense_optical_flow_app
  PUBLIC
    ${CERES_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

target_include_directories(dense_optical_flow_app
  PUBLIC
    ${CERES_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(direct_pose_estimation_app
  ${CMAKE_CURRENT_SOURCE_DIR}/src/DirectPoseEstimationApp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/DirectPoseEstimation.cpp
)

target_link_libraries(direct_pose_estimation_app
  PUBLIC
    ${CERES_LIBRARIES}
    Sophus::Sophus
    ${OpenCV_LIBRARIES}
)

target_include_directories(direct_pose_estimation_app
  PUBLIC
    ${CERES_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
