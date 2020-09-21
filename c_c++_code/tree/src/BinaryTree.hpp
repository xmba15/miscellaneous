/**
 * @file    BinaryTree.hpp
 *
 * @brief   Header for Binary Tree
 *
 * @author  xmba15
 *
 * @date    2019-01-29
 *
 * miscellaneous
 *
 * Copyright (c) organization
 *
 */

#ifndef TREE_HPP_
#define TREE_HPP_

#include "Node.hpp"
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
#include <queue>
#include <sstream>
#include <string>

namespace algo
{
template <class T> void printLevelOrder(typename Node<T>::Ptr root)
{
    if (!root) {
        return;
    }

    std::queue<typename Node<T>::Ptr> queuePtr;
    queuePtr.push(root);

    while (!queuePtr.empty()) {
        typename Node<T>::Ptr node = queuePtr.front();

        std::cout << *node;
        queuePtr.pop();

        if (node->left) {
            queuePtr.push(node->left);
        }

        if (node->right) {
            queuePtr.push(node->right);
        }
    }
}

template <typename T, typename NodeType = Node<T>> class BinaryTree
{
 public:
    enum TRAVERSAL_TYPE { INORDER, PREORDER, POSTORDER };

    using NodePtr = typename NodeType::Ptr;
    using Ptr = std::shared_ptr<BinaryTree>;

    explicit BinaryTree(const NodePtr &rootPtr);

    const NodePtr &rootPtr() const
    {
        return _rootPtr;
    };

    NodePtr &rootPtr()
    {
        return _rootPtr;
    }

    virtual void insert(const NodePtr &nodePtr)
    {
    }

    void traverse(TRAVERSAL_TYPE traversalType = INORDER);
    void traverse(const NodePtr &nodePtr,
                  TRAVERSAL_TYPE traversalType = INORDER);
    bool isBinarySearchTree() const;

    int height(NodePtr node)
    {
        if (!node) {
            return -1;
        }

        return node->height;
    }

    int getBalance(NodePtr node);

    friend std::ostream &operator<<(std::ostream &os,
                                    const BinaryTree<T, NodeType> &bt)
    {
        std::string title = "digraph";
        std::string direction = " -> ";
        std::string firstSpace = "    ";
        os << title << " {\n";

        os << bt.visualizeUtil(bt._rootPtr, direction, firstSpace);

        os << "}";
        return os;
    }

 protected:
    bool isBinarySearchTree(const NodePtr &nodePtr) const;
    bool isBinarySearchTreeUtil(const NodePtr &nodePtr, T min, T max) const;

    void traverseInorder(const NodePtr &nodePtr);    //  Left, Root, Right
    void traversePreorder(const NodePtr &nodePtr);   // Root, Left, Right
    void traversePostorder(const NodePtr &nodePtr);  // Left, Right, Root

    std::string visualizeUtil(const NodePtr &nodePtr,
                              const std::string &direction,
                              const std::string &firstSpace) const
    {
        std::string result;

        if (!nodePtr) {
            return "";
        }

        if (nodePtr->left) {
            result += visualizeUtil(nodePtr->left, direction, firstSpace);
            std::stringstream ss;
            ss << firstSpace << nodePtr->data << direction
               << nodePtr->left->data << ";\n";
            result += ss.str();
        }

        if (nodePtr->right) {
            result += visualizeUtil(nodePtr->right, direction, firstSpace);
            std::stringstream ss;
            ss << firstSpace << nodePtr->data << direction
               << nodePtr->right->data << ";\n";
            result += ss.str();
        }

        return result;
    }

    NodePtr _rootPtr;
};

template <typename T, typename NodeType>
BinaryTree<T, NodeType>::BinaryTree(const NodePtr &rootPtr) : _rootPtr(rootPtr)
{
}

template <typename T, typename NodeType>
bool BinaryTree<T, NodeType>::isBinarySearchTree() const
{
    return isBinarySearchTree(this->_rootPtr);
}

template <typename T, typename NodeType>
bool BinaryTree<T, NodeType>::isBinarySearchTree(const NodePtr &nodePtr) const
{
    return isBinarySearchTreeUtil(nodePtr, std::numeric_limits<T>::min(),
                                  std::numeric_limits<T>::max());
}

template <typename T, typename NodeType>
bool BinaryTree<T, NodeType>::isBinarySearchTreeUtil(const NodePtr &nodePtr,
                                                     T min, T max) const
{
    if (!nodePtr) {
        return true;
    }

    if (nodePtr->data <= min || nodePtr->data >= max) {
        return false;
    }

    return isBinarySearchTreeUtil(nodePtr->left, min, nodePtr->data) &&
           isBinarySearchTreeUtil(nodePtr->right, nodePtr->data, max);
}

template <typename T, typename NodeType>
void BinaryTree<T, NodeType>::traverseInorder(const NodePtr &nodePtr)
{
    if (!nodePtr) {
        return;
    }

    this->traverseInorder(nodePtr->left);

    std::cout << *nodePtr;

    this->traverseInorder(nodePtr->right);
}

template <typename T, typename NodeType>
void BinaryTree<T, NodeType>::traversePreorder(const NodePtr &nodePtr)
{
    if (!nodePtr) {
        return;
    }

    std::cout << *nodePtr;

    this->traversePreorder(nodePtr->left);

    this->traversePreorder(nodePtr->right);
}

template <typename T, typename NodeType>
void BinaryTree<T, NodeType>::traversePostorder(const NodePtr &nodePtr)
{
    if (!nodePtr) {
        return;
    }

    this->traversePostorder(nodePtr->left);

    this->traversePostorder(nodePtr->right);

    std::cout << *nodePtr;
}

template <typename T, typename NodeType>
void BinaryTree<T, NodeType>::traverse(const NodePtr &nodePtr,
                                       TRAVERSAL_TYPE traversalType)
{
    switch (traversalType) {
        case INORDER: {
            this->traverseInorder(nodePtr);
            break;
        }
        case PREORDER: {
            this->traversePreorder(nodePtr);
            break;
        }
        case POSTORDER: {
            this->traversePostorder(nodePtr);
            break;
        }
        default:
            break;
    }
}

template <typename T, typename NodeType>
void BinaryTree<T, NodeType>::traverse(TRAVERSAL_TYPE traversalType)
{
    if (this->_rootPtr) {
        this->traverse(this->_rootPtr, traversalType);
    }
}

template <typename T, typename NodeType>
int BinaryTree<T, NodeType>::getBalance(NodePtr node)
{
    if (!node) {
        return 0;
    }

    return height(node->left) - height(node->right);
}

}  // namespace algo
#endif /* TREE_HPP_ */
