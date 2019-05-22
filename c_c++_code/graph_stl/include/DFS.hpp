/**
 * @file    DFS.hpp
 *
 * @brief   Depth First Search
 *
 * @author  btran
 *
 * @date    2019-05-21
 *
 * Copyright (c) organization
 *
 */

#ifndef DFS_H
#define DFS_H

#include "Search.hpp"
#include <iostream>
#include <memory>
#include <stack>
#include <vector>

namespace algo
{
template <typename T, typename WEIGHT_TYPE = double>
class DFS : public Search<T, WEIGHT_TYPE>
{
 public:
    using Ptr = std::shared_ptr<DFS>;
    using GRAPH_TYPE = Graph<T, WEIGHT_TYPE>;
    using VERTEX_TYPE = typename Graph<T, WEIGHT_TYPE>::Vertex;

    explicit DFS(const typename GRAPH_TYPE::Ptr &graphPtr)
        : Search<T, WEIGHT_TYPE>(graphPtr)
    {
    }

    void visit(const VERTEX_TYPE &v) override;

    void visit() override;

    void showVertexStack(std::stack<VERTEX_TYPE> vertexStack)
    {
        while (!vertexStack.empty()) {
            std::cout << vertexStack.top() << "\n";
            vertexStack.pop();
        }
    }

    void showTopologicalSort()
    {
        this->showVertexStack(this->_topologicalSort);
    }

 protected:
    virtual void visitDirectedGraph(const VERTEX_TYPE &v);
    virtual void visitUndirectedGraph(const VERTEX_TYPE &v);
    std::map<VERTEX_TYPE, int> _numbering;
    int _counter;

 private:
    std::stack<VERTEX_TYPE> _topologicalSort;
};

template <typename T, typename WEIGHT_TYPE>
void DFS<T, WEIGHT_TYPE>::visitUndirectedGraph(const VERTEX_TYPE &v)
{
    this->_visited[v] = true;
    auto it = this->_graphPtr->adjList().find(v);
    if (it != this->_graphPtr->adjList().end()) {
        for (const VERTEX_TYPE &vertex : it->second) {
            if (!this->_visited[vertex]) {
                visit(vertex);
            }
        }
    }
}

template <typename T, typename WEIGHT_TYPE>
void DFS<T, WEIGHT_TYPE>::visitDirectedGraph(const VERTEX_TYPE &v)
{
    this->_numbering[v] = ++_counter;
    this->_visited[v] = true;

    auto it = this->_graphPtr->adjList().find(v);
    if (it != this->_graphPtr->adjList().end()) {
        for (const VERTEX_TYPE &vertex : it->second) {
            std::cout << v << "->" << vertex;
            if (this->_numbering[vertex] == 0) {
                std::cout << ": tree edge\n";
                visit(vertex);
            } else if (this->_numbering[vertex] > this->_numbering[v]) {
                std::cout << ": forward edge\n";
            } else if (this->_visited[vertex]) {
                std::cout << ": back edge\n";
            } else {
                std::cout << ": cross edge\n";
            }
        }
    }
    this->_topologicalSort.push(v);
    this->_visited[v] = false;
}

template <typename T, typename WEIGHT_TYPE>
void DFS<T, WEIGHT_TYPE>::visit(const VERTEX_TYPE &v)
{
    if (!this->isDirected()) {
        this->visitUndirectedGraph(v);
    }

    if (this->isDirected()) {
        this->visitDirectedGraph(v);
    }
}

template <typename T, typename WEIGHT_TYPE> void DFS<T, WEIGHT_TYPE>::visit()
{
    if (!this->_graphPtr) {
        return;
    }

    // Initialization for both directed and undirected graph
    for (const auto &curPair : this->_graphPtr->adjList()) {
        this->_visited[curPair.first] = false;
    }

    // Undirected Graph
    if (!this->isDirected()) {
        for (const auto &curPair : this->_graphPtr->adjList()) {
            if (!this->_visited[curPair.first]) {
                this->visit(curPair.first);
            }
        }
    }

    // Directed Graph
    if (this->isDirected()) {
        // Initialization for directed graph

        for (const auto &curPair : this->_graphPtr->adjList()) {
            this->_numbering[curPair.first] = 0;
        }

        this->_counter = 0;

        for (const auto &curPair : this->_graphPtr->adjList()) {
            if (this->_numbering[curPair.first] == 0) {
                this->visit(curPair.first);
            }
        }
    }
}

}  // namespace algo

#endif /* DFS_H */
