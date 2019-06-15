/**
 * @file    BinarySearchTree.hpp
 *
 * @brief   Header for Binary Search Tree
 *
 * @author  bt
 *
 * @date    2019-06-16
 *
 * Copyright (c) organization
 *
 */

#ifndef BINARYSEARCHTREE_HPP_
#define BINARYSEARCHTREE_HPP_

#include "BinaryTree.hpp"
#include <memory>

namespace algo
{
template <typename T> class BinarySearchTree : public BinaryTree<T>
{
 public:
    using BinaryTree<T>::BinaryTree;
    using NodePtr = typename BinaryTree<T>::NodePtr;
    using Ptr = std::shared_ptr<BinarySearchTree>;

    NodePtr search(const NodePtr &root, T key);
    NodePtr insert(NodePtr node, T key);
};

template <typename T>
typename BinarySearchTree<T>::NodePtr
BinarySearchTree<T>::search(const NodePtr &root, T key)
{
    if (root == nullptr || root->data == key) {
        return root;
    }

    if (root->data < key) {
        return search(root->right, key);
    }

    return search(root->left, key);
}

template <typename T>
typename BinarySearchTree<T>::NodePtr BinarySearchTree<T>::insert(NodePtr node,
                                                                  T key)
{
    if (node == nullptr) {
        node = createNewNode(key);
        return node;
    }

    if (key < node->data) {
        node->left = insert(node->left, key);
    } else if (key > node->data) {
        node->right = insert(node->right, key);
    }

    return node;
}

}  // namespace algo
#endif /* BINARYSEARCHTREE_HPP_ */
