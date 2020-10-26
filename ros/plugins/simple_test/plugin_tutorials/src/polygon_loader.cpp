/**
 * @file    polygon_loader.cpp
 *
 */

#include <plugin_tutorials/polygon_plugins.hpp>
#include <pluginlib/class_loader.h>

int main(int argc, char* argv[])
{
    pluginlib::ClassLoader<geometry::RegularPolygon> polyLoader("plugin_tutorials", "geometry::RegularPolygon");

    try {
        auto triangle = polyLoader.createInstance("geometry::Triangle");
        triangle->initialize(10.0);

        auto square = polyLoader.createInstance("geometry::Square");
        square->initialize(10.0);

        ROS_INFO("Triangle area: %.2f", triangle->area());
        ROS_INFO("Square area: %.2f", square->area());
    } catch (pluginlib::PluginlibException& ex) {
        ROS_ERROR("The plugin failed to load for some reason. Error: %s", ex.what());
    }

    return 0;
}
