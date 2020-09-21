// Copyright (c) 2018
// All Rights Reserved.

#include "maze.hpp"

Maze::Maze(int width, int height) : width(width), height(height) {
  this->data = new cell[width*height];
}

Maze::Maze(int width, int height, cell* data) : width(width), height(height) {
  this->data = data;
}

bool Maze::safeAccess(int row, int col) {
  return ((row >= 0 && row <= this->height) &&
          (col >= 0 && col <= this->width));
}

Maze::~Maze() {}

cell& Maze::getCell(int row, int col) const {
  return data[row * this->width + col];
}

void Maze::setCell(int row, int col, bool status) {
  if (safeAccess(row, col)) getCell(row, col).setVal(status);
}

std::ostream& operator<<(std::ostream& os, Maze& maze) {
  std::string result = "";
  for (int i = 0; i < maze.height; i++) {
    for (int j = 0; j < maze.width; j++) {
      os << maze.data[i * maze.width + j].val << ' ';
    }
    os << '\n';
  }
  return os;
}

bool Maze::isSafe(int row, int col) {
  return safeAccess(row, col) && getCell(row, col).getVal();
}

bool Maze::solMazeUntil(int x, int y, Maze& result) {
  if (x == this->height - 1 && y == this->width - 1) {
    result.getCell(x, y).setVal(true);
    return true;
  }
  if (isSafe(x, y)) {
    result.getCell(x, y).setVal(true);
    if (solMazeUntil(x + 1, y, result)) return true;
    if (solMazeUntil(x, y + 1, result)) return true;
    result.getCell(x, y).setVal(false);
    return false;
  }
  return false;
}
