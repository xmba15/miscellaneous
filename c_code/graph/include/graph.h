// Copyright (c) 2018
// All Rights Reserved.
// Author: btran@btranPC (btran)

#ifndef GRAPH_H
#define GRAPH_H

typedef struct AdjListNode* AdjListNodePtr;
typedef struct AdjList* AdjListPtr;
typedef struct Graph* GraphPtr;

struct AdjListNode {
  int dest;
  AdjListNodePtr next;
};

struct AdjList {
  AdjListNodePtr head;
};

struct Graph {
  int V;
  AdjListPtr array;
};

AdjListNodePtr newAdjListNode(int dest);
AdjListNodePtr getLastElement(AdjListNodePtr);
GraphPtr createGraph(int V);
void printGraph(GraphPtr graph);
void addEdge(GraphPtr graph, int src, int dest);
void printGraph(GraphPtr graph);

#endif /* GRAPH_H */
