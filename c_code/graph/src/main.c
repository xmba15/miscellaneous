// Copyright (c) 2018
// All Rights Reserved.
// Author: btran@btranPC (btran)

#include <stdio.h>
#include "graph.h"

int main(int argc, char *argv[]) {
  int V = 5;
  GraphPtr graph = createGraph(V);
  addEdge(graph, 0, 1);
  addEdge(graph, 0, 4);
  addEdge(graph, 1, 2);
  addEdge(graph, 1, 3);
  addEdge(graph, 1, 4);
  addEdge(graph, 2, 3);
  addEdge(graph, 3, 4);

  printGraph(graph);

  free(graph);

  return 0;
}
