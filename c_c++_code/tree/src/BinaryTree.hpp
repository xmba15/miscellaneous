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
#include <memory>
#include <queue>

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

template <typename T> class BinaryTree
{
 public:
    enum TRAVERSAL_TYPE { INORDER, PREORDER, POSTORDER };

    using NodePtr = typename Node<T>::Ptr;

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

    void traverse(TRAVERSAL_TYPE traversalType = INORDER);
    void traverse(const NodePtr &nodePtr,
                  TRAVERSAL_TYPE traversalType = INORDER);

 private:
    void traverseInorder(const NodePtr &nodePtr);    //  Left, Root, Right
    void traversePreorder(const NodePtr &nodePtr);   // Root, Left, Right
    void traversePostorder(const NodePtr &nodePtr);  // Left, Right, Root

 private:
    NodePtr _rootPtr;
};

template <typename T>
BinaryTree<T>::BinaryTree(const NodePtr &rootPtr) : _rootPtr(rootPtr)
{
}

template <typename T>
void BinaryTree<T>::traverseInorder(const NodePtr &nodePtr)
{
    if (!nodePtr) {
        return;
    }

    this->traverseInorder(nodePtr->left);

    std::cout << *nodePtr;

    this->traverseInorder(nodePtr->right);
}

template <typename T>
void BinaryTree<T>::traversePreorder(const NodePtr &nodePtr)
{
    if (!nodePtr) {
        return;
    }

    std::cout << *nodePtr;

    this->traversePreorder(nodePtr->left);

    this->traversePreorder(nodePtr->right);
}

template <typename T>
void BinaryTree<T>::traversePostorder(const NodePtr &nodePtr)
{
    if (!nodePtr) {
        return;
    }

    this->traversePostorder(nodePtr->left);

    this->traversePostorder(nodePtr->right);

    std::cout << *nodePtr;
}

template <typename T>
void BinaryTree<T>::traverse(const NodePtr &nodePtr,
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

template <typename T> void BinaryTree<T>::traverse(TRAVERSAL_TYPE traversalType)
{
    if (this->_rootPtr) {
        this->traverse(this->_rootPtr, traversalType);
    }
}

}  // namespace algo
#endif /* TREE_HPP_ */
