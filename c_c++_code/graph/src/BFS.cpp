/**
 * @file    BFS.cpp
 *
 * @brief   test BFS graph traversal
 *
 * @author  xmba15
 *
 * @date    2019-01-29
 *
 * miscellaneous
 *
 * Copyright (c) organization
 *
 */

#include "BFS.hpp"

int main(int argc, char *argv[]) {
  Graph g(4);
  g.addEdge(0, 1);
  g.addEdge(0, 2);
  g.addEdge(1, 2);
  g.addEdge(2, 0);
  g.addEdge(2, 3);
  g.addEdge(3, 3);

  // g.BFS(2);
  g.BFS(3);

  return 0;
}
