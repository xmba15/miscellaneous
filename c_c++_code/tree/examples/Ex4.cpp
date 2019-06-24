/**
 * @file    Ex3.cpp
 *
 * @brief   test avl tree
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

#include "AVLTree.hpp"
#include <memory>

using Node = algo::Node<float>;
using AVLTree = algo::AVLTree<float>;
using algo::createNewNode;

int main(int argc, char *argv[])
{
    AVLTree::Ptr avlt = std::make_shared<AVLTree>(nullptr);

    std::vector<float> keyV = {1,   10,     10,  12,  3,  0.19,
                               0.5, 1000.1, 500, 240, 50, 491};

    for (auto key : keyV) {
        avlt->rootPtr() = avlt->insert(avlt->rootPtr(), key);
    }

    avlt->traverse(AVLTree::PREORDER);

    bool isAVLT = avlt->isBinarySearchTree();

    std::cout << "Is this a binary search tree?"
              << "\n";

    std::cout << isAVLT << "\n";

    std::cout << *avlt << "\n";

    std::cout << "delete root"
              << "\n";

    avlt->rootPtr() = avlt->deleteNode(avlt->rootPtr(), 12);

    avlt->traverse(AVLTree::PREORDER);

    std::cout << *avlt << "\n";

    return 0;
}
