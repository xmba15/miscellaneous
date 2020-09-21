/**
 * @file    plugin_nodelet_talker.hpp
 */

#pragma once

#include <string>

#include <nodelet/nodelet.h>
#include <ros/ros.h>

namespace plugin_lecture
{
class plugin_nodelet_talker : public nodelet::Nodelet
{
 public:
    virtual void onInit();
    void timer_callback(const ros::TimerEvent&);

 private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Publisher pub_;
    ros::Timer timer_;
    std::string content_;
};
}  // namespace plugin_lecture
