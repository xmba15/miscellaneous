/**
 * @file    polygon_plugins.hpp
 *
 */

#pragma once

#include <cmath>

#include "polygon_base.hpp"

namespace geometry
{
class Triangle : public geometry::RegularPolygon
{
 public:
    void initialize(double side_length)
    {
        side_length_ = side_length;
    }

    double area()
    {
        return 0.5 * side_length_ * getHeight();
    }

    double getHeight()
    {
        return std::sqrt((side_length_ * side_length_) - ((side_length_ / 2) * (side_length_ / 2)));
    }

 private:
    double side_length_;
};

class Square : public geometry::RegularPolygon
{
 public:
    void initialize(double side_length)
    {
        side_length_ = side_length;
    }

    double area()
    {
        return side_length_ * side_length_;
    }

 private:
    double side_length_;
};
}  // namespace geometry
