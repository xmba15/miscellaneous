cmake_minimum_required(VERSION 3.10)

project(integral_image LANGUAGES CXX)
set(APP_NAME ${PROJECT_NAME}_app)

set (CMAKE_CXX_STANDARD 17)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else(CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA not found")
  return()
endif()

find_package(Threads REQUIRED)

add_executable(${APP_NAME}
  ${PROJECT_SOURCE_DIR}/src/App.cpp
)

target_link_libraries(${APP_NAME}
  Threads::Threads
)

set_source_files_properties(
  ${PROJECT_SOURCE_DIR}/src/App.cpp
    PROPERTIES LANGUAGE CUDA
)

target_compile_options(${APP_NAME}
  INTERFACE
     $<$<CONFIG:Debug>:-O0 -g -Wall -Werror>
     $<$<CONFIG:Release>:-O3>
  PRIVATE
     $<$<COMPILE_LANGUAGE:CUDA>:-expt-extended-lambda>
)
