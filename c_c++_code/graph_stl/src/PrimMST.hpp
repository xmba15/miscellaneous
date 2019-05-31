/**
 * @file    PrimMST.hpp
 *
 * @brief   Implementation of Prim's Minimum Spanning Tree
 *
 * @author  btran
 *
 * @date    2019-05-31
 *
 * Copyright (c) organization
 *
 */

#ifndef PRIMMST_HPP_
#define PRIMMST_HPP_

#include "Search.hpp"
#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace algo
{
template <typename T, typename WEIGHT_TYPE = double>
class PrimMST : public Search<T, WEIGHT_TYPE>
{
 public:
    using Ptr = std::shared_ptr<PrimMST>;
    using GRAPH_TYPE = Graph<T, WEIGHT_TYPE>;
    using Vertex = typename Graph<T, WEIGHT_TYPE>::Vertex;
    using Edge = typename GRAPH_TYPE::Edge;

    // map a vertex with its parent vertex in the MST
    using PrimMSTMap = std::map<Vertex, Vertex *>;

    explicit PrimMST(const typename GRAPH_TYPE::Ptr &graphPtr)
        : Search<T, WEIGHT_TYPE>(graphPtr)
    {
    }

    std::vector<Edge> findMST();
};

template <typename T, typename WEIGHT_TYPE>
std::vector<typename PrimMST<T, WEIGHT_TYPE>::Edge>
PrimMST<T, WEIGHT_TYPE>::findMST()
{
    using DistanceNode = std::pair<WEIGHT_TYPE, Vertex>;
    std::vector<Edge> edgesInMST;

    // store vertex's distance
    std::map<Vertex, WEIGHT_TYPE> vDistMap;

    // store vertex's prev vertex in an edge
    std::map<Vertex, Vertex> vPrevMap;

    std::map<Vertex, bool> visited;

    auto adjList = this->_graphPtr->adjList();

    // set U to first store all vertex
    std::priority_queue<DistanceNode, std::vector<DistanceNode>,
                        std::greater<DistanceNode>>
        minHeap;

    for (auto it = adjList.begin(); it != adjList.end(); ++it) {
        vDistMap.insert(
            std::make_pair(it->first, std::numeric_limits<WEIGHT_TYPE>::max()));
        visited.insert(std::make_pair(it->first, false));
    }

    // using the first vertex element as the root of the MST
    const Vertex &rootMST = adjList.begin()->first;

    vDistMap[rootMST] = 0.0;
    vPrevMap.insert(std::make_pair(rootMST, rootMST));

    minHeap.push(std::make_pair(0.0, rootMST));

    while (!minHeap.empty()) {
        auto frontPair = minHeap.top();
        minHeap.pop();
        Vertex curV = frontPair.second;
        visited[curV] = true;

        for (const Vertex &adjV : adjList[curV]) {
            Edge curEdge = std::make_pair(curV, adjV);

            WEIGHT_TYPE curWeight = this->_graphPtr->weights(curEdge);

            if (!visited[adjV] && vDistMap[adjV] > curWeight) {
                vDistMap[adjV] = curWeight;

                vPrevMap.insert(std::make_pair(adjV, curV));
                minHeap.push(std::make_pair(curWeight, adjV));
            }
        }
    }

    // fill the MST vector edges
    for (auto pairV : vPrevMap) {
        if (pairV.second != pairV.first) {
            edgesInMST.push_back(std::make_pair(pairV.second, pairV.first));
        }
    }

    return edgesInMST;
}

}  // namespace algo

#endif /* PRIMMST_HPP_ */
