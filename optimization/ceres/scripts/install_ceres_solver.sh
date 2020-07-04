#!/usr/bin/env bash

sudo -l

# Ref: http://ceres-solver.org/installation.html

sudo apt-get -y install libgoogle-glog-dev
sudo apt-get -y install libatlas-base-dev
sudo apt-get -y install libeigen3-dev

# build ceres as a shared library
sudo add-apt-repository -y ppa:bzindovic/suitesparse-bugfix-1319687
sudo apt-get -y update
sudo apt-get -y install libsuitesparse-dev

readonly CERES_VERSION="1.14.0"
git clone -b ${CERES_VERSION} --recursive https://github.com/ceres-solver/ceres-solver
cd ceres-solver
mkdir -p build && cd build && cmake ../ && make -j`nproc`
sudo make install
