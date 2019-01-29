/**
 * @file    main.cpp
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

#include "tree.hpp"
#include <memory>

int main(int argc, char *argv[]) {
  std::unique_ptr<Node<float>> rootPtr(createNewNode<float>(1));
  rootPtr.get()->left = createNewNode<float>(2);
  rootPtr.get()->right = createNewNode<float>(3);
  rootPtr.get()->left->left = createNewNode<float>(4);
  rootPtr.get()->left->right = createNewNode<float>(5);
  printLevelOrder<float>(rootPtr.get());

  return 0;
}
