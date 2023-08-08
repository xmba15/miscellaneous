```bash
ODE_VERSION="0.16.4"
git clone -b ${ODE_VERSION} --recursive https://bitbucket.org/odedevs/ode.git

# change the newline in the following lines in CMakeLists.txt into -
# set(CPACK_COMPONENT_DEVELOPMENT_DESCRIPTION "Open Dynamics Engine - development files-${CPACK_DEBIAN_PACKAGE_DESCRIPTION}")
# set(CPACK_COMPONENT_RUNTIME_DESCRIPTION "Open Dynamics Engine - runtime library-${CPACK_DEBIAN_PACKAGE_DESCRIPTION}")

mkdir ode_build && && cd $_
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
ninja -j`nproc`
```
