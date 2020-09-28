/**
 * @file    joystick_teleop_node.cpp
 *
 * @author  btran
 *
 */

#include <ackermann_msgs/AckermannDriveStamped.h>

#include "joystick_teleop_node.hpp"

JoystickTeleopHandler::JoystickTeleopHandler(ros::NodeHandle& nh, ros::NodeHandle& pnh)

{
    pnh.param<std::string>("frame_id", m_param.frameId, m_param.frameId);

    m_joySub = pnh.subscribe<sensor_msgs::Joy>("input_joy", 10, &JoystickTeleopHandler::joyCallback, this);
    m_drivePub = pnh.advertise<ackermann_msgs::AckermannDriveStamped>("output_cmd", 1);
}

JoystickTeleopHandler::~JoystickTeleopHandler()
{
}

void JoystickTeleopHandler::joyCallback(const sensor_msgs::Joy::ConstPtr& joyMsg)
{
    ackermann_msgs::AckermannDriveStamped cmdMsg;
    cmdMsg.header.stamp = ros::Time::now();
    cmdMsg.header.frame_id = m_param.frameId;
    cmdMsg.drive.acceleration = 1;
    cmdMsg.drive.jerk = 1;
    cmdMsg.drive.steering_angle_velocity = 1;
    cmdMsg.drive.speed = joyMsg->axes[2] * m_param.maxSpeed;
    cmdMsg.drive.steering_angle = joyMsg->axes[3] * m_param.maxSteeringAngle;
    m_drivePub.publish(cmdMsg);
}

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "joystick_teleop");

    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    JoystickTeleopHandler obj(nh, pnh);

    ros::spin();

    return 0;
}
