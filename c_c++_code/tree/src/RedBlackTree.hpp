/**
 * @file    RedBlackTree.hpp
 *
 * @brief   Header for Red Black Tree (Self Balance BST)
 *
 * @author  bt
 *
 * @date    2019-06-16
 *
 * Copyright (c) organization
 *
 */

#ifndef REDBLACKTREE_HPP_
#define REDBLACKTREE_HPP_

#include "BinarySearchTree.hpp"

namespace algo
{
template <typename T, typename NodeType = Node<T>>
class RedBlackTree : public BinarySearchTree<T, NodeType>
{
 public:
    using Ptr = std::shared_ptr<RedBlackTree>;
    using NodePtr = typename BinarySearchTree<T>::NodePtr;

    explicit RedBlackTree(const NodePtr &rootPtr)
        : BinarySearchTree<T, NodeType>(rootPtr)
    {
    }

    NodePtr insert(NodePtr node, T key) override;
    NodePtr deleteNode(NodePtr node, T key) override;
};

template <typename T, typename NodeType>
typename RedBlackTree<T, NodeType>::NodePtr
RedBlackTree<T, NodeType>::insert(NodePtr node, T key)
{
    return nullptr;
}

template <typename T, typename NodeType>
typename RedBlackTree<T, NodeType>::NodePtr
RedBlackTree<T, NodeType>::deleteNode(NodePtr node, T key)
{
    return nullptr;
}

}  // namespace algo
#endif /* REDBLACKTREE_HPP_ */
