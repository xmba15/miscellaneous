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

    // rbt->rootPtr() = rbt->insert(rbt->rootPtr(), 10);

    // rbt->rootPtr() = rbt->insert(rbt->rootPtr(), 10);
    // rbt->rootPtr() = rbt->insert(rbt->rootPtr(), 12);

    // rbt->insert(rbt->rootPtr(), 3);
    // rbt->insert(rbt->rootPtr(), 0.5);

    // rbt->traverse();

    // bool isRBT = rbt->isBinarySearchTree();

    // std::cout << "Is this a binary search tree?"
    //           << "\n";

    // std::cout << isRBT << "\n";

    // std::cout << *rbt << "\n";

    return 0;
}
