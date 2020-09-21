/**
 * @file    Example3.cpp
 *
 * @brief   test Topological Sort
 *
 * @author  btran
 *
 * @date    2019-05-21
 *
 * Copyright (c) organization
 *
 */

#include "Graph.hpp"
#include <TopologicalSortKahn.hpp>
#include <iostream>

using Graph = algo::Graph<int>;
using TopologicalSortKahn = algo::TopologicalSortKahn<int>;

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

    Graph::Ptr g2 = std::make_shared<Graph>(*g);
    g2->addEdge(4, 2);

    TopologicalSortKahn::Ptr tsk = std::make_shared<TopologicalSortKahn>(g);
    if (tsk->doTopologicalSort()) {
        tsk->printSortedVertices();
    }

    TopologicalSortKahn::Ptr tsk2 = std::make_shared<TopologicalSortKahn>(g2);
    if (tsk2->doTopologicalSort()) {
        tsk2->printSortedVertices();
    } else {
        std::cout << "None found"
                  << "\n";
    }

    return 0;
}
