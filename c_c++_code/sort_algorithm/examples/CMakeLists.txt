set(EX1_NAME "InsertionSort")
set(EX2_NAME "MergeSort")

build_example(
  ${EX1_NAME}
  ${CMAKE_CURRENT_LIST_DIR}/Example1.cpp
  sort_algorithm
)

build_example(
  ${EX2_NAME}
  ${CMAKE_CURRENT_LIST_DIR}/Example2.cpp
  sort_algorithm
)
