add_executable(
  Ex1
  ${CMAKE_CURRENT_LIST_DIR}/Ex1.cpp
  )

target_link_libraries(
  Ex1
  test_tree
  )
