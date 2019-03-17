cmake_minimum_required(VERSION 3.0)

project(test_pyobject)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
add_definitions("-Wall")

find_package(PythonLibs 2 QUIET)
if(NOT ${PythonLibs_FOUND})
  execute_process(COMMAND python-config --prefix OUTPUT_VARIABLE PYTHON_SEARCH_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REGEX REPLACE "\n$" "" PYTHON_SEARCH_PATH "${PYTHON_SEARCH_PATH}")
  file(GLOB_RECURSE PYTHON_STATIC_LIBS ${PYTHON_SEARCH_PATH}/lib/libpython*.so)

  if (PYTHON_STATIC_LIBS)
    list(GET PYTHON_STATIC_LIBS 0 PYTHON_LIBRARY)
    message(STATUS "Found static python libs")
    execute_process(COMMAND python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())" OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
    # python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
  else()
    message(WARNING "Cannot find libpython")
    exit()
  endif()
endif()

file(GLOB BASIC_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  )

add_executable(${PROJECT_NAME}
  ${BASIC_SOURCE_FILES}
  )

target_include_directories(${PROJECT_NAME}
  PUBLIC ${PYTHON_INCLUDE_DIRS}
)

message(STATUS "${PYTHON_LIBRARY}")
# target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARY})
