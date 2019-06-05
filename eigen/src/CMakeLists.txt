set(LIBRARY_NAME
  test_eigen
)

file(GLOB SOURCE_FILES
  ${CMAKE_CURRENT_LIST_DIR}/*.cpp
)

add_library(
  ${LIBRARY_NAME}
  SHARED
  ${SOURCE_FILES}
)

target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(
  ${LIBRARY_NAME}
  Eigen3::Eigen
)
