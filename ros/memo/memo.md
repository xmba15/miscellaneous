# Installation #

```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | sudo apt-key add -
sudo apt update
echo "install full version"
sudo apt install ros-melodic-desktop-full
```

# Build from source #
```bash
mkdir ~/ros_catkin_ws
cd ~/ros_catkin_ws
# install gazebo manually
curl -sSL http://get.gazebosim.org | sh


rosinstall_generator desktop_full --rosdistro melodic --deps --tar > melodic-desktop-full.rosinstall
wstool init -j8 src melodic-desktop-full.rosinstall
sudo rosdep init
rosdep update

# use bionic dep
rosdep install --from-paths src --ignore-src --rosdistro=${ROS_DISTRO} -y --os=ubuntu:bionic

./src/catkin/bin/catkin_make_isolated --install -DCMAKE_BUILD_TYPE=Release -DSETUPTOOLS_DEB_LAYOUT=OFF \
    -DPYTHON_EXECUTABLE:FILEPATH=`which python` \
    -DPYTHON_LIBRARIES=$(python-config --prefix)/lib/libpython3.6m.so.1.0 \
    -DPYTHON_INCLUDE_DIRS=$(python-config --prefix)/include/python3.6m
```
# QA #
- there might be problem with Qt
- pip install empy
