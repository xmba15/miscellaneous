add_executable(
  DFS
  ${CMAKE_CURRENT_LIST_DIR}/Example1.cpp
)

target_link_libraries(
  DFS
  graph_stl
  )

add_executable(
  BFS
  ${CMAKE_CURRENT_LIST_DIR}/Example2.cpp
)

target_link_libraries(
  BFS
  graph_stl
  )

add_executable(
  Topological_Sort_Kahn
  ${CMAKE_CURRENT_LIST_DIR}/Example3.cpp
  )

target_link_libraries(
  Topological_Sort_Kahn
  graph_stl
  )
