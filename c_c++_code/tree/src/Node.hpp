/**
 * @file    Node.hpp
 *
 * @brief   Header for Node
 *
 * @author  btran
 *
 * @date    2019-06-11
 *
 * Copyright (c) organization
 *
 */

#ifndef NODE_HPP_
#define NODE_HPP_

#include <memory>
#include <ostream>

namespace algo
{
template <typename T> struct Node {
    using Ptr = std::shared_ptr<Node>;
    T data;
    Ptr left, right;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const Node<T> &node)
{
    os << node.data << "\n";
    return os;
}

template <typename T> typename Node<T>::Ptr createNewNode(T data)
{
    typename Node<T>::Ptr node = std::make_shared<Node<T>>();
    node->data = data;
    node->left = nullptr;
    node->right = nullptr;

    return node;
}

}  // namespace algo
#endif /* NODE_HPP_ */
