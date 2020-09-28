/**
 * @file    joystick_teleop_node.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <string>

#include <ros/ros.h>
#include <sensor_msgs/Joy.h>

class JoystickTeleopHandler
{
 public:
    struct Param {
        std::string frameId = "base_link";
        double maxSpeed = 0.5;          // [m/s]
        double maxSteeringAngle = 0.7;  // [rad]
    };

 public:
    JoystickTeleopHandler(ros::NodeHandle& nh, ros::NodeHandle& pnh);
    virtual ~JoystickTeleopHandler();

 private:
    void joyCallback(const sensor_msgs::Joy::ConstPtr& joyMsg);

 private:
    Param m_param;

    ros::Subscriber m_joySub;
    ros::Publisher m_drivePub;
};
