set(EX1_NAME "DFS")
set(EX2_NAME "BFS")
set(EX3_NAME "Topological_Sort_Kahn")
set(EX4_NAME "PrimMST")

build_example(
  ${EX1_NAME}
  ${CMAKE_CURRENT_LIST_DIR}/Example1.cpp
  graph_stl
)

build_example(
  ${EX2_NAME}
  ${CMAKE_CURRENT_LIST_DIR}/Example2.cpp
  graph_stl
)

build_example(
  ${EX3_NAME}
  ${CMAKE_CURRENT_LIST_DIR}/Example3.cpp
  graph_stl
)

build_example(
  ${EX4_NAME}
  ${CMAKE_CURRENT_LIST_DIR}/Example4.cpp
  graph_stl
)
