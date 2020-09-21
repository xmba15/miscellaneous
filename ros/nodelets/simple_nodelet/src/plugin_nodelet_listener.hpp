/**
 * @file    plugin_nodelet_listener.hpp
 *
 */

#pragma once

#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <std_msgs/String.h>

namespace plugin_lecture
{
class plugin_nodelet_listener : public nodelet::Nodelet
{
 public:
    virtual void onInit();
    void chatter_callback(const std_msgs::String& string_msg);

 private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
};
}  // namespace plugin_lecture
