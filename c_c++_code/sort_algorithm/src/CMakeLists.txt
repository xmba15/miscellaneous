set(LIBRARY_NAME sort_algorithm)

add_library(
  ${LIBRARY_NAME}
  INTERFACE
)

target_include_directories(
  ${LIBRARY_NAME}
  INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}
)
