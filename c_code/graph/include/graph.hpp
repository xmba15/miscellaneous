// Copyright (c) 2018
// All Rights Reserved.
// Author: btran@btranPC (btran)

#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <memory>

typedef std::shared_ptr<struct AdjListNode> AdjListNodePtr;
typedef std::shared_ptr<struct AdjList> AdjListPtr;
typedef std::shared_ptr<struct Graph> GraphPtr;

typedef struct AdjListNode {
  int dest;
  AdjListNodePtr next;
} AdjListNode;


typedef struct AdjList {
  AdjListNodePtr head;
} AdjList;

typedef struct Graph {
  int V;
  AdjListPtr array;
} Graph;

AdjListNodePtr newAdjListNode(int dest);
AdjListNodePtr getLastElement(AdjListNodePtr node);
GraphPtr createGraph(int V);
void addEdge(GraphPtr graph, int src, int dest);
void printGraph(GraphPtr graph);

#endif /* GRAPH_HPP */
