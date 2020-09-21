set(EX1_NAME "BinaryTree")
set(EX2_NAME "BinarySearchTree")
set(EX3_NAME "RedBlackTree")
set(EX4_NAME "AVLTree")

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

build_example(
  ${EX3_NAME}
  ${CMAKE_CURRENT_LIST_DIR}/Ex3.cpp
  tree_algorithm
)

build_example(
  ${EX4_NAME}
  ${CMAKE_CURRENT_LIST_DIR}/Ex4.cpp
  tree_algorithm
)
