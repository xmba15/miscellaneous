cmake_minimum_required(VERSION 3.10)

project(simple_gpu_image_processing LANGUAGES CXX)
set(LIBRARY_NAME ${PROJECT_NAME})

set (CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else(CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA not found")
  return()
endif()

add_executable(convert_to_gray
  ${PROJECT_SOURCE_DIR}/src/ConvertToGrayApp.cpp
)
target_link_libraries(convert_to_gray
  PUBLIC
    ${OpenCV_LIBRARIES}
)
target_include_directories(convert_to_gray
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)
set_source_files_properties(
  ${PROJECT_SOURCE_DIR}/src/ConvertToGrayApp.cpp
    PROPERTIES LANGUAGE CUDA
)

# --------------------------------------------------------------------------------

add_executable(convolve_image
  ${PROJECT_SOURCE_DIR}/src/ImageConvolutionApp.cpp
)
target_link_libraries(convolve_image
  PUBLIC
    ${OpenCV_LIBRARIES}
)
target_include_directories(convolve_image
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)
set_source_files_properties(
  ${PROJECT_SOURCE_DIR}/src/ImageConvolutionApp.cpp
    PROPERTIES LANGUAGE CUDA
)

# --------------------------------------------------------------------------------

find_package(CUDA REQUIRED)

add_executable(dft1f
  ${PROJECT_SOURCE_DIR}/src/DFT1FApp.cpp
)
target_link_libraries(dft1f
  PUBLIC
    ${OpenCV_LIBRARIES}
    ${CUDA_cufft_LIBRARY}
)
target_include_directories(dft1f
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)
set_source_files_properties(
  ${PROJECT_SOURCE_DIR}/src/DFT1FApp.cpp
    PROPERTIES LANGUAGE CUDA
)

# --------------------------------------------------------------------------------

add_executable(histogram
  ${PROJECT_SOURCE_DIR}/src/HistogramApp.cpp
)
target_link_libraries(histogram
  PUBLIC
    ${OpenCV_LIBRARIES}
)
target_include_directories(histogram
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)
set_source_files_properties(
  ${PROJECT_SOURCE_DIR}/src/HistogramApp.cpp
    PROPERTIES LANGUAGE CUDA
)
