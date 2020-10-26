/**
 * @file    plugin_nodelet_talker.cpp
 */

#include <pluginlib/class_list_macros.h>
#include <std_msgs/String.h>

#include "plugin_nodelet_talker.hpp"

namespace plugin_lecture
{
void plugin_nodelet_talker::onInit()
{
    NODELET_INFO("Talker Init");
    nh_ = getNodeHandle();
    pnh_ = getPrivateNodeHandle();
    content_ = "hello";
    pnh_.getParam("content", content_);
    pub_ = nh_.advertise<std_msgs::String>("chatter", 10);
    timer_ = nh_.createTimer(ros::Duration(1.0), &plugin_nodelet_talker::timer_callback, this);
}

void plugin_nodelet_talker::timer_callback(const ros::TimerEvent&)
{
    NODELET_INFO("send: %s", content_.c_str());
    std_msgs::String string_msg;
    string_msg.data = content_;
    pub_.publish(string_msg);
}
}  // namespace plugin_lecture

PLUGINLIB_EXPORT_CLASS(plugin_lecture::plugin_nodelet_talker, nodelet::Nodelet);
