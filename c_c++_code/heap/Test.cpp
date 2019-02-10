/**
 * @file    Test.cpp
 *
 * @brief   test file
 *
 * @author  xmba15
 *
 * @date    2019-01-30
 *
 * miscellaneous framework
 *
 * Copyright (c) organization
 *
 */

#include "MaxHeap.hpp"
#include <iostream>
#include <vector>

int main(int argc, char *argv[])
{
  MaxHeap<double> maxHeap;
  maxHeap.insertKey(3);
  maxHeap.insertKey(5);
  maxHeap.deleteKey(1);
  maxHeap.insertKey(15.2);
  maxHeap.insertKey(5);
  maxHeap.insertKey(4);
  maxHeap.insertKey(45.8);

  maxHeap.increaseKey(2, 50);

  maxHeap.BFSTraversal(0);
  maxHeap.writeGraph("outputGraph");

  return 0;
}
