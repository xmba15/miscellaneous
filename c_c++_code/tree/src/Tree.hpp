/**
 * @file    Tree.hpp
 *
 * @brief   test tree traversal using queue
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

#include <iostream>
#include <memory>
#include <queue>

template <typename T> struct Node {
    using Ptr = std::shared_ptr<Node>;
    T data;
    Ptr left, right;
};

template <typename T> typename Node<T>::Ptr createNewNode(T data)
{
    typename Node<T>::Ptr temp = std::make_shared<Node<T>>();
    temp->data = data;
    temp->left = nullptr;
    temp->right = nullptr;
    return temp;
}

template <class T> void printLevelOrder(typename Node<T>::Ptr root)
{
    if (!root) {
        return;
    }

    std::queue<typename Node<T>::Ptr> queuePtr;
    queuePtr.push(root);

    while (!queuePtr.empty()) {
        typename Node<T>::Ptr node = queuePtr.front();

        std::cout << node->data << "\n";
        queuePtr.pop();

        if (node->left) {
            queuePtr.push(node->left);
        }

        if (node->right) {
            queuePtr.push(node->right);
        }
    }
}

#endif /* TREE_HPP_ */
