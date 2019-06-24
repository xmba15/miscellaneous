/**
 * @file    AVLTree.hpp
 *
 * @brief   Header for AVL Tree (Self Balance BST)
 *
 * @author  bt
 *
 * @date    2019-06-16
 *
 * Copyright (c) organization
 *
 */

#ifndef AVLTREE_HPP_
#define AVLTREE_HPP_

#include "BinarySearchTree.hpp"
#include <algorithm>

namespace algo
{
template <typename T, typename NodeType = Node<T>>
class AVLTree : public BinarySearchTree<T, NodeType>
{
 public:
    using BinarySearchTree<T, NodeType>::BinarySearchTree;
    using Ptr = std::shared_ptr<AVLTree>;
    using NodePtr = typename BinarySearchTree<T>::NodePtr;

    NodePtr insert(NodePtr node, T key) override;
    NodePtr deleteNode(NodePtr node, T key) override;
};

template <typename T, typename NodeType>
typename AVLTree<T, NodeType>::NodePtr
AVLTree<T, NodeType>::insert(NodePtr node, T key)
{
    if (node == nullptr) {
        return createNewNode<T, NodeType>(key);
    }

    if (key < node->data) {
        node->left = insert(node->left, key);
    } else if (key > node->data) {
        node->right = insert(node->right, key);
    } else {
        return node;
    }

    node->height =
        1 + std::max(this->height(node->left), this->height(node->right));

    int balance = this->getBalance(node);

    // Left Left Case
    if (balance > 1 && key < node->left->data) {
        return this->rightRotate(node);
    }

    // Right Right Case
    if (balance < -1 && key > node->right->data) {
        return this->leftRotate(node);
    }

    // Left Right Case
    if (balance > 1 && key > node->left->data) {
        node->left = this->leftRotate(node->left);
        return this->rightRotate(node);
    }

    // Right Left Case
    if (balance < -1 && key < node->right->data) {
        node->right = this->rightRotate(node->right);
        return this->leftRotate(node);
    }

    return node;
}

template <typename T, typename NodeType>
typename AVLTree<T, NodeType>::NodePtr
AVLTree<T, NodeType>::deleteNode(NodePtr node, T key)
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
            node.reset(node->right.get());
            return node;
        } else if (!node->right) {
            node.reset(node->left.get());
            return node;
        } else {
            NodePtr inorderSuccessor = this->minValueNode(node->right);
            node->data = inorderSuccessor->data;
            node->right = deleteNode(node->right, inorderSuccessor->data);
        }
    }

    if (!node) {
        return node;
    }

    node->height =
        1 + std::max(this->height(node->left), this->height(node->right));

    int balance = this->getBalance(node);

    // Left Left Case
    if (balance > 1 && this->getBalance(node->left) >= 0) {
        return this->rightRotate(node);
    }

    // Right Right Case
    if (balance < -1 && this->getBalance(node->right) <= 0) {
        return this->leftRotate(node);
    }

    // Left Right Case
    if (balance > 1 && this->getBalance(node->left) < 0) {
        node->left = this->leftRotate(node->left);
        return this->rightRotate(node);
    }

    // Right Left Case
    if (balance < -1 && this->getBalance(node->right) > 0) {
        node->right = this->rightRotate(node->right);
        return this->leftRotate(node);
    }

    return node;
}

}  // namespace algo
#endif /* AVLTREE_HPP_ */
