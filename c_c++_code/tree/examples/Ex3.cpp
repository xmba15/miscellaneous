/**
 * @file    Ex3.cpp
 *
 * @brief   test red black tree
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

#include "RedBlackTree.hpp"
#include <memory>

using Node = algo::Node<float>;
using RedBlackTree = algo::RedBlackTree<float>;
using algo::createNewNode;

int main(int argc, char *argv[])
{
    Node::Ptr rootPtr = createNewNode<float, Node>(1);

    RedBlackTree::Ptr rbt = std::make_shared<RedBlackTree>(rootPtr);

    // bst->insert(bst->rootPtr(), 10);
    // bst->insert(bst->rootPtr(), 10);
    // bst->insert(bst->rootPtr(), 12);
    // bst->insert(bst->rootPtr(), 3);
    // bst->insert(bst->rootPtr(), 0.5);

    // bst->traverse();

    // bool isBST = bst->isBinarySearchTree()();

    // std::cout << "Is this a binary search tree?"
    // << "\n";
    // std::cout << isBST << "\n";

    return 0;
}
