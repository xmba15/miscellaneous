/**
 * @file    Graph.hpp
 *
 * @brief   Generic Graph Type
 *
 * @author  btran
 *
 * @date    2019-05-17
 *
 * Copyright (c) organization
 *
 */

#ifndef GRAPH_HPP_
#define GRAPH_HPP_

#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <utility>

namespace algo
{
template <typename T, typename WEIGHT_TYPE = double> class Graph
{
 public:
    class Vertex;

    using Edge = std::pair<Vertex, Vertex>;

    using Ptr = std::shared_ptr<Graph>;

    explicit Graph(bool isDirected = false, bool isWeighted = false)
        : _isDirected(isDirected), _isWeighted(isWeighted)
    {
    }

    void addEdge(const T &source, const T &dest, WEIGHT_TYPE weight = 0.0);

    const std::map<Vertex, std::set<Vertex>> &adjList() const
    {
        return this->_adjList;
    }

    const std::map<Edge, WEIGHT_TYPE> &weights() const
    {
        return this->_weights;
    }

    int const &v() const;

    int const &e() const;

    bool const &isDirected() const
    {
        return this->_isDirected;
    }

    bool const &isWeighted() const
    {
        return this->_isWeighted;
    }

    const std::map<Vertex, int> &indegrees() const
    {
        return this->_indegrees;
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const Graph<T, WEIGHT_TYPE> &g)
    {
        std::string graphTitle = g.isDirected() ? "digraph" : "graph";
        std::string direction = g.isDirected() ? " -> " : " -- ";
        std::string firstSpace = "    ";
        os << graphTitle << " {\n";

        for (const auto &curPair : g.adjList()) {
            for (const auto &vertex : curPair.second) {
                os << firstSpace << curPair.first.info() << direction
                   << vertex.info() << ";\n";
            }
        }

        os << "}";
        return os;
    }

 private:
    void initializeIndegree(const Vertex &v);

    bool _isDirected;
    bool _isWeighted;

    std::map<Vertex, int> _indegrees;

    // adjacency list
    std::map<Vertex, std::set<Vertex>> _adjList;

    std::map<Edge, WEIGHT_TYPE> _weights;
};

template <typename T, typename WEIGHT_TYPE> class Graph<T, WEIGHT_TYPE>::Vertex
{
 public:
    explicit Vertex(T info) : _info(info)
    {
    }

    ~Vertex()
    {
    }

    const T &info() const
    {
        return this->_info;
    }

    bool operator==(const Vertex &other) const
    {
        return this->_info == other._info;
    }

    bool operator<(const Vertex &other) const
    {
        return this->_info < other._info;
    }

    friend std::ostream &operator<<(std::ostream &os, const Vertex &v)
    {
        os << v.info();
        return os;
    }

 private:
    T _info;
};

template <typename T, typename WEIGHT_TYPE>
void Graph<T, WEIGHT_TYPE>::initializeIndegree(const Vertex &v)
{
    if (this->_indegrees.find(v) == this->_indegrees.end()) {
        this->_indegrees[v] = 0;
    }
}

template <typename T, typename WEIGHT_TYPE>
void Graph<T, WEIGHT_TYPE>::addEdge(const T &source, const T &dest,
                                    WEIGHT_TYPE weight)
{
    const Vertex srcVertex(source);
    const Vertex destVertex(dest);

    this->_adjList[srcVertex].emplace(destVertex);

    // set indegrees
    this->initializeIndegree(destVertex);
    this->initializeIndegree(srcVertex);
    this->_indegrees[destVertex]++;
    if (!this->_isDirected) {
        this->_indegrees[srcVertex]++;
    }

    if (!this->_isDirected) {
        this->_adjList[destVertex].emplace(srcVertex);
    } else if (this->_adjList.count(destVertex) == 0) {
        this->_adjList[destVertex] = std::set<Vertex>();
    }

    if (this->_isWeighted) {
        this->_weights[std::make_pair(srcVertex, destVertex)] = weight;
        if (!this->_isDirected) {
            this->_weights[std::make_pair(destVertex, srcVertex)] = weight;
        }
    }
}

template <typename T, typename WEIGHT_TYPE>
int const &Graph<T, WEIGHT_TYPE>::v() const
{
    return this->_adjList.size();
}

template <typename T, typename WEIGHT_TYPE>
int const &Graph<T, WEIGHT_TYPE>::e() const
{
    return this->_isDirected ? this->_weights.size()
                             : this->_weights.size() / 2;
}

}  // namespace algo

#endif /* GRAPH_HPP_ */
