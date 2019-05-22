/**
 * @file    TopologicalSortKahn.hpp
 *
 * @brief   Topological Sort Implementation Using Kahn's Algorithm
 *
 * @author  btran
 *
 * @date    2019-05-21
 *
 * Copyright (c) organization
 *
 */

/* Kahn's Algorithm:
Source: https://www.techiedelight.com/kahn-topological-sort-algorithm/
L-> Empty list that will contain the sorted elements
S-> Set of all vertices with no incoming edges (i.e. having indegree 0)

while S is non-empty do
  remove a vertex n from S
  add n to tail of L
  for each vertex m with an edge e from n to m do
    remove edge e from the graph
    if m has no ther incoming edges then
      insert m into S

if graph has edges then
  return report "graph has at least one cycle"
else
  return L "a topological sorted order"

Algorithms for finding all the possile topological orderings
https://www.techiedelight.com/find-all-possible-topological-orderings-of-dag/
*/

#ifndef TOPOLOGICALSORTKAHN_HPP_
#define TOPOLOGICALSORTKAHN_HPP_

#include "Search.hpp"
#include <iostream>
#include <vector>

namespace algo
{
template <typename T, typename WEIGHT_TYPE = double>
class TopologicalSortKahn : public Search<T, WEIGHT_TYPE>
{
 public:
    using Ptr = std::shared_ptr<TopologicalSortKahn>;
    using GRAPH_TYPE = Graph<T, WEIGHT_TYPE>;
    using VERTEX_TYPE = typename Graph<T, WEIGHT_TYPE>::Vertex;

    explicit TopologicalSortKahn(const typename GRAPH_TYPE::Ptr &graphPtr)
        : Search<T, WEIGHT_TYPE>(graphPtr)
    {
    }

    bool doTopologicalSort();

    const std::vector<VERTEX_TYPE> &sortedVertices()
    {
        return this->_sortedVertices;
    }

    void printSortedVertices()
    {
        for (const auto &v : this->_sortedVertices) {
            std::cout << v << "\n";
        }
    }

 private:
    std::vector<VERTEX_TYPE> _sortedVertices;
};

template <typename T, typename WEIGHT_TYPE>
bool TopologicalSortKahn<T, WEIGHT_TYPE>::doTopologicalSort()
{
    std::vector<VERTEX_TYPE> zeroIndegVertices;
    std::map<VERTEX_TYPE, int> indegrees = this->_graphPtr->indegrees();

    for (const auto &vertexIndegPair : indegrees) {
        if (vertexIndegPair.second == 0) {
            zeroIndegVertices.push_back(vertexIndegPair.first);
        }
    }

    while (!zeroIndegVertices.empty()) {
        VERTEX_TYPE curV = zeroIndegVertices.back();
        zeroIndegVertices.pop_back();
        this->_sortedVertices.push_back(curV);

        auto it = this->_graphPtr->adjList().find(curV);

        for (VERTEX_TYPE childV : it->second) {
            indegrees[childV]--;

            if (indegrees[childV] == 0) {
                zeroIndegVertices.push_back(childV);
            }
        }
    }

    for (const auto &vertexIndegPair : indegrees) {
        if (vertexIndegPair.second) {
            return false;
        }
    }

    return true;
}

}  // namespace algo

#endif /* TOPOLOGICALSORTKAHN_HPP_ */
