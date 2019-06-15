/**
 * @file    Ex1.cpp
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

#include "BinaryTree.hpp"
#include <memory>

using Node = algo::Node<float>;
using BinaryTree = algo::BinaryTree<float>;
using algo::createNewNode;

int main(int argc, char *argv[])
{
    Node::Ptr rootPtr = createNewNode<float>(1);

    rootPtr.get()->left = createNewNode<float>(2);
    rootPtr.get()->right = createNewNode<float>(3);
    rootPtr.get()->left->left = createNewNode<float>(4);
    rootPtr.get()->left->right = createNewNode<float>(5);

    BinaryTree::Ptr binaryTree = std::make_shared<BinaryTree>(rootPtr);

    binaryTree->traverse(BinaryTree::POSTORDER);

    bool isBinarySearchTree = binaryTree->isBinarySearchTree();

    std::cout << "Is this a binary search tree?" << "\n";
    std::cout << isBinarySearchTree << "\n";

    std::cout << *binaryTree << "\n";

    return 0;
}
