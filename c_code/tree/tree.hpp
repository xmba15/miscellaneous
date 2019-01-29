/**
 * @file    tree.hpp
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

#include <queue>
#include <iostream>

template <class T>
struct Node
{
  T data;
  struct Node<T> *left, *right;
};

template <class T>
Node<T> *createNewNode(T data) {
  Node<T> *temp = new Node<T>;
  temp->data = data;
  temp->left = nullptr;
  temp->right = nullptr;
  return temp;
}

template <class T>
void printLevelOrder(Node<T> *root)
{
  if (root == nullptr) {
    return;
  }

  std::queue<Node<T> *> q;
  q.push(root);
  while (!q.empty())
  {
    Node<T> *node = q.front();
    std::cout << node->data << "\n";
    q.pop();
    if (node->left != nullptr) {
      q.push(node->left);
    }
    if (node->right != nullptr) {
      q.push(node->right);
    }
  }
}

#endif /* TREE_HPP_ */
