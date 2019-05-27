/**
 * @file    BFS.hpp
 *
 * @brief   Breath First Search
 *
 * @author  btran
 *
 * @date    2019-05-21
 *
 * Copyright (c) organization *
 */

#ifndef BFS_H
#define BFS_H

#include "Search.hpp"
#include <iostream>
#include <queue>

namespace algo
{
template <typename T, typename WEIGHT_TYPE = double>
class BFS : public Search<T, WEIGHT_TYPE>
{
 public:
    using Ptr = std::shared_ptr<BFS>;
    using GRAPH_TYPE = Graph<T, WEIGHT_TYPE>;
    using VERTEX_TYPE = typename Graph<T, WEIGHT_TYPE>::Vertex;

    explicit BFS(const typename GRAPH_TYPE::Ptr &graphPtr)
        : Search<T, WEIGHT_TYPE>(graphPtr)
    {
    }

    void visit(const VERTEX_TYPE &v) override;

    void visit() override;
};

template <typename T, typename WEIGHT_TYPE>
void BFS<T, WEIGHT_TYPE>::visit(const VERTEX_TYPE &v)
{
    std::queue<VERTEX_TYPE> vertexQueue;

    this->_visited[v] = true;
    auto it = this->_graphPtr->adjList().find(v);
    if (it != this->_graphPtr->adjList().end()) {
        vertexQueue.push(it->first);
    }

    while (!vertexQueue.empty()) {
        VERTEX_TYPE curV = vertexQueue.front();
        std::cout << curV << "\n";
        vertexQueue.pop();

        auto it = this->_graphPtr->adjList().find(curV);
        if (it != this->_graphPtr->adjList().end()) {
            for (const VERTEX_TYPE &vertex : it->second) {
                if (!this->_visited[vertex]) {
                    this->_visited[vertex] = true;
                    vertexQueue.push(vertex);
                }
            }
        }
    }
}

template <typename T, typename WEIGHT_TYPE> void BFS<T, WEIGHT_TYPE>::visit()
{
    if (!this->_graphPtr) {
        return;
    }

    // Initialization for both directed and undirected graph
    for (const auto &curPair : this->_graphPtr->adjList()) {
        this->_visited[curPair.first] = false;
    }

    for (const auto &curPair : this->_graphPtr->adjList()) {
        if (!this->_visited[curPair.first]) {
            this->visit(curPair.first);
        }
    }
}

}  // namespace algo

#endif /* BFS_H */
