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
    enum { RED, BLACK } color;
    using Ptr = std::shared_ptr<Node>;
    T data;
    int height;
    Ptr left, right, parent;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const Node<T> &node)
{
    os << node.data << "\n";
    return os;
}

template <typename T, typename NodeType = Node<T>>
typename NodeType::Ptr createNewNode(T data)
{
    typename NodeType::Ptr node = std::make_shared<NodeType>();
    node->data = data;
    node->left = nullptr;
    node->right = nullptr;
    node->parent = nullptr;
    int height = 0;

    return node;
}

// for convenience, Node struct holds all the variales needed to implement Red
// Black Tree, AVL Tree.
template <typename T> struct RBTNode {
    enum { RED, BLACK } color;
    using Ptr = std::shared_ptr<RBTNode>;
    T data;
    Ptr parent, left, right;
};

template <typename T> struct AVLNode {
    using Ptr = std::shared_ptr<AVLNode>;
    T data;
    int height;
    Ptr left, right;
};

}  // namespace algo
#endif /* NODE_HPP_ */
