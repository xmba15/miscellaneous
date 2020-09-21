// Copyright (c) 2018
// All Rights Reserved.
// Author: btran@btranPC (btran)

#include <stdio.h>
#include <stdlib.h>
#include "graph.h"

AdjListNodePtr newAdjListNode(int dest) {
  AdjListNodePtr newNode =
      (AdjListNodePtr) malloc(sizeof(struct AdjListNode));
  newNode->dest = dest;
  newNode->next = NULL;
  return newNode;
}

AdjListNodePtr getLastElement(AdjListNodePtr node) {
  if (node == NULL)
    return NULL;
  AdjListNodePtr cur = node;
  while (cur->next != NULL) {
    cur = cur->next;
  }
  return cur;
}

GraphPtr createGraph(int V) {
  GraphPtr graph = (GraphPtr) malloc(sizeof(struct Graph));
  graph->V = V;
  graph->array = (AdjListPtr) malloc(V * sizeof(struct AdjList));
  for (int i = 0; i < V; ++i) {
    graph->array[i].head = NULL;
  }
  return graph;
}

void addEdge(GraphPtr graph, int src, int dest) {
  AdjListNodePtr newNode = newAdjListNode(dest);
  if (graph->array[src].head == NULL) {
    graph->array[src].head = newNode;
  } else {
    AdjListNodePtr lastElem = getLastElement(graph->array[src].head);
    lastElem->next = newNode;
  }

  newNode = newAdjListNode(src);
  if (graph->array[dest].head == NULL) {
    graph->array[dest].head = newNode;
  } else {
    AdjListNodePtr lastElem = getLastElement(graph->array[dest].head);
    lastElem->next = newNode;
  }
}

void printGraph(GraphPtr graph) {
  for (int i = 0; i < graph->V; ++i) {
    printf("Adjaceny list of vertex %d\n head", i);
    for (AdjListNodePtr cur = graph->array[i].head;
         cur != NULL; cur = cur->next) {
      printf("-> %d", cur->dest);
    }
    printf("\n");
  }
}
