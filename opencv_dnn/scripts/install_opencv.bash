#!/usr/bin/env bash

# change this according to gpu arch
readonly USE_CUDA_ARCH=6.1
readonly OPENCV_DNN_MIN_VERSION=4.2
readonly OPENCV_INSTALL_PREFIX=/usr/local

git clone -b ${OPENCV_DNN_MIN_VERSION} https://github.com/opencv/opencv.git
git clone -b ${OPENCV_DNN_MIN_VERSION} https://github.com/opencv/opencv_contrib.git

cd opencv && mkdir -p build
cmake ../ -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_PREFIX} \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D BUILD_SHARED_LIBS=ON \
      -D WITH_TBB=ON \
      -D BUILD_TBB=ON \
      -D WITH_EIGEN=ON \
      -D WITH_LAPACK=ON \
      -D WITH_GTK=ON \
      -D WITH_OPENGL=ON \
      -D WITH_OPENCL=ON \
      -D WITH_CUDA=ON \
      -D ENABLE_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D WITH_CUBLAS=ON \
      -D CUDA_ARCH_BIN=${USE_CUDA_ARCH} \
      -D CUDA_ARCH_PTX=${USE_CUDA_ARCH}

make -j `nproc`
sudo make install -j `nproc`
