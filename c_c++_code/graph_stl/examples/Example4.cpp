/**
 * @file    Example4.cpp
 *
 * @brief   Example for Prim Minimum Spanning Tree
 *
 * @author  btran
 *
 * @date    2019-05-31
 *
 * Copyright (c) organization
 *
 */

#include "Graph.hpp"
#include "PrimMST.hpp"
#include <iostream>

using Graph = algo::Graph<int>;
using PrimMST = algo::PrimMST<int>;

int main(int argc, char *argv[])
{
    const bool isDirected = false;
    const bool isWeighted = true;

    Graph::Ptr g = std::make_shared<Graph>(isDirected, isWeighted);

    g->addEdge(1, 2, 14.0);
    g->addEdge(1, 5, 3.5);
    g->addEdge(1, 8, 1.2);
    g->addEdge(2, 3, 8.7);
    g->addEdge(5, 6, 12.3);
    g->addEdge(3, 4, 3.9);
    g->addEdge(6, 3, 20.1);
    g->addEdge(6, 7, 21.0);
    g->addEdge(6, 8, 13.7);

    std::cout << *g << "\n\n";

    PrimMST::Ptr primMSTPtr = std::make_shared<PrimMST>(g);

    auto mstEdges = primMSTPtr->findMST();

    std::cout << "Edges that belong to the minimum spanning tree: " << "\n";
    for  (auto edge : mstEdges) {
      std::cout << edge.first << " " << edge.second << "\n";
    }

    return 0;
}
