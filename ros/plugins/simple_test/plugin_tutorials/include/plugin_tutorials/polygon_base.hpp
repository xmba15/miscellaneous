/**
 * @file    polygon_base.hpp
 *
 */

#pragma once

namespace geometry
{
class RegularPolygon
{
 public:
    virtual void initialize(double side_length) = 0;
    virtual double area() = 0;

 protected:
    RegularPolygon() = default;
};
}  // namespace geometry
