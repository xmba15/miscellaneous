/**
 * @file    Search.hpp
 *
 * @brief   Search.hpp
 *
 * @author  btran
 *
 * @date    2019-05-21
 *
 * Copyright (c) organization
 *
 */

#ifndef SEARCH_H
#define SEARCH_H

#include "Graph.hpp"

namespace algo
{
template <typename T, typename WEIGHT_TYPE = double> class Search
{
 public:
    using Ptr = std::shared_ptr<Search>;
    using GRAPH_TYPE = Graph<T, WEIGHT_TYPE>;

    using VERTEX_TYPE = typename Graph<T, WEIGHT_TYPE>::Vertex;

    explicit Search(const typename GRAPH_TYPE::Ptr &graphPtr);

    const bool isDirected() const
    {
        return this->_graphPtr->isDirected();
    }

    virtual void visit(const VERTEX_TYPE &v)
    {
    }

    virtual void visit()
    {
    }

 protected:
    std::map<VERTEX_TYPE, bool> _visited;

    typename GRAPH_TYPE::Ptr _graphPtr;
};

template <typename T, typename WEIGHT_TYPE>
Search<T, WEIGHT_TYPE>::Search(const typename GRAPH_TYPE::Ptr &graphPtr)
    : _graphPtr(graphPtr)
{
}

}  // namespace algo

#endif /* SEARCH_H */
