cmake_minimum_required(VERSION 3.10)

project(stereo_vision)
set (CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(OpenMP QUIET)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(OpenCV REQUIRED)

list(APPEND COMMON_FILES
  ${PROJECT_SOURCE_DIR}/src/StereoEngine.cpp
)

add_executable(${PROJECT_NAME}
  ${PROJECT_SOURCE_DIR}/src/App.cpp
  ${PROJECT_SOURCE_DIR}/src/BlockMatching.cpp
  ${PROJECT_SOURCE_DIR}/src/SemiGlobalMatching.cpp
  ${COMMON_FILES}
)
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    ${OpenCV_LIBRARIES}
    stdc++fs
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)
