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
#include <algorithm>
#include <memory>

namespace algo
{
template <typename T, typename NodeType = Node<T>>
class BinarySearchTree : public BinaryTree<T, NodeType>
{
 public:
    using BinaryTree<T, NodeType>::BinaryTree;
    using NodePtr = typename BinaryTree<T>::NodePtr;
    using Ptr = std::shared_ptr<BinarySearchTree>;

    NodePtr search(const NodePtr &root, T key);
    virtual NodePtr insert(NodePtr node, T key);
    virtual NodePtr deleteNode(NodePtr node, T key);
    NodePtr minValueNode(NodePtr node);

    // rotations
    NodePtr rightRotate(NodePtr node);
    NodePtr leftRotate(NodePtr node);
};

template <typename T, typename NodeType>
typename BinarySearchTree<T, NodeType>::NodePtr
BinarySearchTree<T, NodeType>::search(const NodePtr &root, T key)
{
    if (root == nullptr || root->data == key) {
        return root;
    }

    if (root->data < key) {
        return search(root->right, key);
    }

    return search(root->left, key);
}

template <typename T, typename NodeType>
typename BinarySearchTree<T, NodeType>::NodePtr
BinarySearchTree<T, NodeType>::insert(NodePtr node, T key)
{
    if (node == nullptr) {
        node = createNewNode<T, NodeType>(key);
        return node;
    }

    if (key < node->data) {
        node->left = insert(node->left, key);
    } else if (key > node->data) {
        node->right = insert(node->right, key);
    }

    node->height =
        1 + std::max(this->height(node->left), this->height(node->right));

    return node;
}

template <typename T, typename NodeType>
typename BinarySearchTree<T, NodeType>::NodePtr
BinarySearchTree<T, NodeType>::minValueNode(NodePtr node)
{
    NodePtr current = node;
    while (current && current->left) {
        current = current->left;
    }

    return current;
}

template <typename T, typename NodeType>
typename BinarySearchTree<T, NodeType>::NodePtr
BinarySearchTree<T, NodeType>::deleteNode(NodePtr node, T key)
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

template <typename T, typename NodeType>
typename BinarySearchTree<T, NodeType>::NodePtr
BinarySearchTree<T, NodeType>::rightRotate(NodePtr node)
{
    /**
               node
          x
             y
     **/

    NodePtr x = node->left;
    NodePtr y = x->right;

    x->right = node;
    node->left = y;

    node->height =
        1 + std::max(this->height(node->left), this->height(node->right));
    x->height = 1 + std::max(this->height(x->left), this->height(x->right));

    return x;
}

template <typename T, typename NodeType>
typename BinarySearchTree<T, NodeType>::NodePtr
BinarySearchTree<T, NodeType>::leftRotate(NodePtr node)
{
    /**
               node
                        x
                      y
     **/

    NodePtr x = node->right;
    NodePtr y = x->left;

    x->left = node;
    node->right = y;

    node->height =
        1 + std::max(this->height(node->left), this->height(node->right));
    x->height = 1 + std::max(this->height(x->left), this->height(x->right));

    return x;
}

}  // namespace algo
#endif /* BINARYSEARCHTREE_HPP_ */
