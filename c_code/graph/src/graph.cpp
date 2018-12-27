// Copyright (c) 2018
// All Rights Reserved.
// Author: btran@btranPC (btran)

#include "graph.hpp"

AdjListNodePtr newAdjListNode(int dest) {
  auto newNode = AdjListNodePtr();
  newNode->dest = dest;
  newNode->next = nullptr;
  return newNode;
}

AdjListNodePtr getLastElement(AdjListNodePtr node) {
  if (node == nullptr) {
    return nullptr;
  }
  AdjListNodePtr cur = node;
  while (cur->next != nullptr) {
    cur = cur->next;
  }
  return cur;
}

GraphPtr createGraph(int V) {
  GraphPtr graph = GraphPtr();
  graph->V = V;
  graph->array = AdjListPtr(new AdjList[V], std::default_delete<AdjList[]>());
  for (int i = 0; i < V; ++i) {
    graph->array.get()[i].head = nullptr;
  }
  return graph;
}

// void printGraph(std::shared_ptr<Graph> graph);
// void addEdge(std::shared_ptr<Graph> graph, int src, int dest);
// void printGraph(std::shared_ptr<Graph> graph);
