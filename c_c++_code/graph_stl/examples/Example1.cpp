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
#include <string>

using Graph = algo::Graph<int>;
using StrGraph = algo::Graph<std::string>;

int main(int argc, char *argv[])
{
    {
        bool isDirected = true;
        bool isWeighted = false;
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
        std::cout << *g << "\n";
    }

    {
        bool isDirected = false;
        bool isWeighted = false;
        StrGraph::Ptr g2 = std::make_shared<StrGraph>(isDirected, isWeighted);

        g2->addEdge("a", "b");
        g2->addEdge("a", "e");
        g2->addEdge("a", "h");
        g2->addEdge("b", "c");
        g2->addEdge("e", "f");
        g2->addEdge("c", "d");
        g2->addEdge("f", "c");
        g2->addEdge("f", "g");
        g2->addEdge("f", "h");
        g2->addEdge("d", "b");

        std::cout << *g2 << "\n";
    }

    return 0;
}
