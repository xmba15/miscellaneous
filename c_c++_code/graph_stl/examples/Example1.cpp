/**
 * @file    Example1.cpp
 *
 * @brief   Example1 file
 *
 * @author  btran
 *
 * @date    2019-05-21
 *
 * Copyright (c) organization
 *
 */

#include "Graph.hpp"
#include <iostream>

using Graph = algo::Graph<int>;

int main(int argc, char *argv[])
{
    const bool isDirected = true;
    const bool isWeighted = false;
    Graph g(isDirected, isWeighted);

    g.addEdge(1, 2);
    g.addEdge(1, 5);
    g.addEdge(1, 8);
    g.addEdge(2, 3);
    g.addEdge(5, 6);
    g.addEdge(3, 4);
    g.addEdge(6, 3);
    g.addEdge(6, 7);
    g.addEdge(6, 8);
    g.addEdge(4, 2);

    std::cout << g << "\n";
    return 0;
}
