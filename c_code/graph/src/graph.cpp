// Copyright (c) 2018
// All Rights Reserved.
// Author: btran@btranPC (btran)

#include <iostream>
#include "graph.hpp"

AdjListNodePtr newAdjListNode(int dest) {
  AdjListNodePtr newNode = std::make_shared<AdjListNode>();
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
  GraphPtr graph = std::make_shared<Graph>();
  graph->V = V;
  graph->array = AdjListPtr(new AdjList[V], std::default_delete<AdjList[]>());
  for (int i = 0; i < V; ++i) {
    graph->array.get()[i].head = nullptr;
  }
  return graph;
}

void addEdge(std::shared_ptr<Graph> graph, int src, int dest) {
  AdjListNodePtr newNode = newAdjListNode(dest);
  if (graph->array.get()[src].head == nullptr) {
    graph->array.get()[src].head = newNode;
  } else {
    AdjListNodePtr lastElem = getLastElement(graph->array.get()[src].head);
    lastElem->next = newNode;
  }

  newNode = newAdjListNode(src);
  if (graph->array.get()[dest].head == nullptr) {
    graph->array.get()[dest].head = newNode;
  } else {
    AdjListNodePtr lastElem = getLastElement(graph->array.get()[dest].head);
    lastElem->next = newNode;
  }
}

void printGraph(std::shared_ptr<Graph> graph) {
  for (int i = 0; i < graph->V; ++i) {
    std::cout << "Adjaceny list of vertex " << i << "\n head";
    for (AdjListNodePtr cur = graph->array.get()[i].head;
         cur != nullptr; cur = cur->next) {
      std::cout << "-> " << cur->dest;
    }
    std::cout << "\n";
  }
}
