/**
 * @file    Ex2.cpp
 *
 * @brief   test binary search tree
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

#include "BinarySearchTree.hpp"
#include <memory>

using Node = algo::Node<float>;
using BinarySearchTree = algo::BinarySearchTree<float>;
using algo::createNewNode;

int main(int argc, char *argv[])
{
    Node::Ptr rootPtr = createNewNode<float>(1);

    BinarySearchTree::Ptr bst = std::make_shared<BinarySearchTree>(rootPtr);
    bst->insert(bst->rootPtr(), 10);
    bst->insert(bst->rootPtr(), 10);
    bst->insert(bst->rootPtr(), 12);
    bst->insert(bst->rootPtr(), 3);
    bst->insert(bst->rootPtr(), 0.5);

    bst->traverse();

    bool isBinarySearchTree = bst->isBinarySearchTree();

    std::cout << "Is this a binary search tree?"
              << "\n";
    std::cout << isBinarySearchTree << "\n";

    std::cout << *bst << "\n";

    bst->deleteNode(bst->rootPtr(), 1);

    std::cout << *bst << "\n";

    return 0;
}
