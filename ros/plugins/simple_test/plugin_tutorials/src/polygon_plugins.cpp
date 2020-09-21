/**
 * @file    polygon_plugins.cpp
 */

#include <plugin_tutorials/polygon_plugins.hpp>
#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(geometry::Triangle, geometry::RegularPolygon)
PLUGINLIB_EXPORT_CLASS(geometry::Square, geometry::RegularPolygon)
