/**
 * @file    Test.cpp
 *
 * @brief   test tiny kaboom
 *
 * @author  xmba15
 *
 * @date    2019-01-31
 *
 * miscellaneous framework
 *
 * Copyright (c) organization
 *
 */

#include "Geometry.hpp"
#include <fstream>
#include <vector>

void render()
{
  const int width = 1024;
  const int height = 768;

  vec<width * height, Vec3f> framebuffer;

  for (size_t j = 0; j < height; ++j) {
    for (size_t i = 0; i < width; ++i) {
      framebuffer[i + j * width] = Vec3f(j / static_cast<float>(height),
                                          i / static_cast<float>(width), 0);
    }
  }

  std::ofstream ofs;
  ofs.open("./out.ppm");
  ofs << "P6\n" << width << " " << height << "\n255\n";
  for (size_t i = 0; i < height * width; ++i) {
    for (size_t j = 0; j < 3; j++) {
      ofs << static_cast<char>(
          255 *
          std::max(0.f, std::min(1.f, framebuffer[i][j])));
    }
  }
  ofs.close();
}

int main(int argc, char *argv[])
{
  render();
  return 0;
}
