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
    NodePtr deleteNode(NodePtr node, T key);
    NodePtr minValueNode(NodePtr node);
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

template <typename T>
typename BinarySearchTree<T>::NodePtr
BinarySearchTree<T>::minValueNode(NodePtr node)
{
    NodePtr current = node;
    while (current && current->left) {
        current = current->left;
    }

    return current;
}

template <typename T>
typename BinarySearchTree<T>::NodePtr
BinarySearchTree<T>::deleteNode(NodePtr node, T key)
{
    if (!node) {
        return node;
    }

    if (key < node->data) {
        node->left = deleteNode(node->left, key);
    } else if (key > node->data) {
        node->right = deleteNode(node->right, key);
    } else {
        if (!node->left) {
            // reset pointer in current node to point to the right child
            node.reset(node->right.get());
            return node;
        } else if (!node->right) {
            node.reset(node->left.get());
            return node;
        } else {
            NodePtr inorderSuccessor = minValueNode(node->right);
            node->data = inorderSuccessor->data;
            node->right = deleteNode(node->right, inorderSuccessor->data);
        }
    }

    return node;
}

}  // namespace algo
#endif /* BINARYSEARCHTREE_HPP_ */
