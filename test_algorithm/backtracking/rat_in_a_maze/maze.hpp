// Copyright (c) 2018
// All Rights Reserved.

#ifndef MAZE_H
#define MAZE_H

#include <string>
#include <ostream>

struct cell {
  bool val;
  cell() : val(false) {};
  cell(bool status) : val(status) {};

  bool getVal() {
    return this->val;
  }

  void setVal(bool val) {
    this->val = val;
  }
};

class Maze {
 public:
  Maze(int width, int height);
  Maze(int width, int height, cell* data);
  ~Maze();
  bool safeAccess(int row, int col);

  cell& getCell(int row, int col) const;

  void setCell(int row, int col, bool status);

  bool isSafe(int row, int col);

  bool solMazeUntil(int x, int y, Maze& result);

  friend std::ostream& operator<<(std::ostream& os, Maze& maze);

 private:
  int width;
  int height;
  cell * data;
};

#endif /* MAZE_H */
