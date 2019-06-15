set(EX1_NAME "BinaryTree")
set(EX2_NAME "BinarySearchTree")

build_example(
  ${EX1_NAME}
  ${CMAKE_CURRENT_LIST_DIR}/Ex1.cpp
  tree_algorithm
)

build_example(
  ${EX2_NAME}
  ${CMAKE_CURRENT_LIST_DIR}/Ex2.cpp
  tree_algorithm
)
