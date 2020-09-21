cmake_minimum_required(VERSION 3.10)

project(test_cuda LANGUAGES CUDA CXX)
set (CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
cuda_select_nvcc_arch_flags(ARCH_FLAGS)

if(CUDA_FOUND)
  message(STATUS "Cuda version: ${CUDA_VERSION}")
  add_subdirectory(examples/hello_world)
  add_subdirectory(examples/julia_set)
  add_subdirectory(examples/matrix_add)
  add_subdirectory(examples/shared_memory)
  add_subdirectory(examples/vec_double)

else(CUDA_FOUND)
  message(STATUS "Cuda not found")
endif(CUDA_FOUND)
