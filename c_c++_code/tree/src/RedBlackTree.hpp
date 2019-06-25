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

 protected:
    NodePtr rightRotate(NodePtr node) override;
    NodePtr leftRotate(NodePtr node) override;

    NodePtr grandparent(NodePtr node);
    NodePtr sibling(NodePtr node);
    NodePtr uncle(NodePtr node);

 private:
    NodePtr BSTInsert(NodePtr node, NodePtr nodeToInSert);
    void fixViolation(NodePtr nodeToInsert);

    void fixCase1(NodePtr nodeToInsert);
    void fixCase2(NodePtr nodeToInsert);
    void fixCase3(NodePtr nodeToInsert);
    void fixCase4(NodePtr nodeToInsert);
    void fixCase4Step2(NodePtr nodeToInsert);
};

template <typename T, typename NodeType>
typename RedBlackTree<T, NodeType>::NodePtr
RedBlackTree<T, NodeType>::BSTInsert(NodePtr node, NodePtr nodeToInsert)
{
    if (!node) {
        return nodeToInsert;
    }

    if (nodeToInsert->data < node->data) {
        node->left = BSTInsert(node->left, nodeToInsert);
        node->left->parent = node;
    } else if (nodeToInsert->data > node->data) {
        node->right = BSTInsert(node->right, nodeToInsert);
        node->right->parent = node;
    }

    return node;
}

template <typename T, typename NodeType>
void RedBlackTree<T, NodeType>::fixViolation(NodePtr nodeToInsert)
{
    NodePtr uncleNode = this->uncle(nodeToInsert);

    if (!nodeToInsert->parent) {
        this->fixCase1(nodeToInsert);
    } else if (nodeToInsert->parent->color == NodeType::BLACK) {
        this->fixCase2(nodeToInsert);
    } else if (uncleNode && uncleNode->color == NodeType::RED) {
        this->fixCase3(nodeToInsert);
    } else {
        this->fixCase4(nodeToInsert);
    }
}

template <typename T, typename NodeType>
void RedBlackTree<T, NodeType>::fixCase1(NodePtr nodeToInsert)
{
    if (!nodeToInsert->parent) {
        nodeToInsert->color = NodeType::BLACK;
    }
}

template <typename T, typename NodeType>
void RedBlackTree<T, NodeType>::fixCase2(NodePtr nodeToInsert)
{
    return;
}

template <typename T, typename NodeType>
void RedBlackTree<T, NodeType>::fixCase3(NodePtr nodeToInsert)
{
    nodeToInsert->parent->color = NodeType::BLACK;
    this->uncle(nodeToInsert)->color = NodeType::BLACK;
    this->grandparent(nodeToInsert)->color = NodeType::RED;
    this->fixViolation(this->grandparent(nodeToInsert));
}

template <typename T, typename NodeType>
void RedBlackTree<T, NodeType>::fixCase4(NodePtr nodeToInsert)
{
    NodePtr grandparentNode = this->grandparent(nodeToInsert);
    NodePtr parentNode = nodeToInsert->parent;

    // Left Right
    if (parentNode == grandparentNode->left &&
        nodeToInsert == parentNode->right) {
        parentNode = this->leftRotate(parentNode);
    } else if (parentNode == grandparentNode->right &&
               nodeToInsert == parentNode->left) {
        parentNode = this->rightRotate(parentNode);
    }

    fixCase4Step2(nodeToInsert);
}

template <typename T, typename NodeType>
void RedBlackTree<T, NodeType>::fixCase4Step2(NodePtr nodeToInsert)
{
    NodePtr grandparentNode = this->grandparent(nodeToInsert);
    NodePtr parentNode = nodeToInsert->parent;

    if (nodeToInsert == parentNode->left) {
        grandparentNode = this->rightRotate(grandparentNode);
    } else {
        grandparentNode = this->leftRotate(grandparentNode);
    }

    parentNode->color = NodeType::BLACK;
    grandparentNode->color = NodeType::RED;
}

template <typename T, typename NodeType>
typename RedBlackTree<T, NodeType>::NodePtr
RedBlackTree<T, NodeType>::insert(NodePtr node, T key)
{
    NodePtr nodeToInsert = createNewNode(key);

    node = this->BSTInsert(node, nodeToInsert);

    this->fixViolation(nodeToInsert);

    node = nodeToInsert;
    while (node->parent) {
        node = node->parent;
    }

    return node;
}

template <typename T, typename NodeType>
typename RedBlackTree<T, NodeType>::NodePtr
RedBlackTree<T, NodeType>::deleteNode(NodePtr node, T key)
{
    return nullptr;
}

template <typename T, typename NodeType>
typename RedBlackTree<T, NodeType>::NodePtr
RedBlackTree<T, NodeType>::rightRotate(NodePtr node)
{
    /**
               node
          x
             y
     **/

    NodePtr parentNode = node->parent;
    NodePtr x = node->left;
    NodePtr y = x->right;

    x->right = node;
    x->parent = parentNode;

    if (parentNode) {
        if (node == parentNode->left) {
            parentNode->left = x;
        } else {
            parentNode->right = x;
        }
    }

    node->parent = x;

    node->left = y;
    if (y) {
        y->parent = node;
    }

    return x;
}

template <typename T, typename NodeType>
typename RedBlackTree<T, NodeType>::NodePtr
RedBlackTree<T, NodeType>::leftRotate(NodePtr node)
{
    /**
               node
                        x
                      y
     **/

    NodePtr parentNode = node->parent;
    NodePtr x = node->right;
    NodePtr y = x->left;

    x->left = node;
    x->parent = node->parent;

    if (parentNode) {
        if (node == parentNode->left) {
            parentNode->left = x;
        } else {
            parentNode->right = x;
        }
    }

    node->parent = x;

    node->right = y;
    if (y) {
        y->parent = node;
    }

    return x;
}

template <typename T, typename NodeType>
typename RedBlackTree<T, NodeType>::NodePtr
RedBlackTree<T, NodeType>::grandparent(NodePtr node)
{
    if (!node->parent) {
        return nullptr;
    }

    return node->parent->parent;
}

template <typename T, typename NodeType>
typename RedBlackTree<T, NodeType>::NodePtr
RedBlackTree<T, NodeType>::sibling(NodePtr node)
{
    NodePtr parentNode = node->parent;

    if (!parentNode) {
        return nullptr;
    }

    if (node == parentNode->left) {
        return parentNode->right;
    } else {
        return parentNode->left;
    }
}

template <typename T, typename NodeType>
typename RedBlackTree<T, NodeType>::NodePtr
RedBlackTree<T, NodeType>::uncle(NodePtr node)
{
    NodePtr parentNode = node->parent;
    NodePtr grandparentNode = parentNode->parent;

    if (!grandparentNode) {
        return nullptr;
    }

    return this->sibling(parentNode);
}

}  // namespace algo
#endif /* REDBLACKTREE_HPP_ */
