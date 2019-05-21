/**
 * @file    Example3.cpp
 *
 * @brief   test BFS
 *
 * @author  btran
 *
 * @date    2019-05-21
 *
 * Copyright (c) organization
 *
 */

#include "BFS.hpp"
#include "Graph.hpp"
#include <iostream>

using Graph = algo::Graph<int>;
using BFS = algo::BFS<int>;

int main(int argc, char *argv[])
{
    const bool isDirected = true;
    const bool isWeighted = false;
    Graph::Ptr g = std::make_shared<Graph>(isDirected, isWeighted);

    g->addEdge(1, 2);
    g->addEdge(1, 5);
    g->addEdge(1, 8);
    g->addEdge(2, 3);
    g->addEdge(5, 6);
    g->addEdge(3, 4);
    g->addEdge(6, 3);
    g->addEdge(6, 7);
    g->addEdge(6, 8);
    g->addEdge(4, 2);
    g->addEdge(100, 1000);

    BFS::Ptr bfs = std::make_shared<BFS>(g);

    bfs->visit();

    return 0;
}
