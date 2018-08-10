// Copyright (c) 2018
// All Rights Reserved.

#include <iostream>
#include "maze.hpp"

int main(int argc, char *argv[]) {

  cell data[] = {1, 0, 0, 0,
                 1, 1, 0, 1,
                 0, 1, 0, 0,
                 1, 1, 1, 1};

  Maze maze(4, 4, data);
  Maze solution(4, 4);
  maze.solMazeUntil(0, 0, solution);
  std::cout << maze << '\n';
  std::cout << solution << '\n';

  return 0;
}
