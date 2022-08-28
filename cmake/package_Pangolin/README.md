# 📝 example of packaging Pangolin with cpack
***

## :running: How to Run ##
***

- packing with cpack

```bash
export ROOT_DIR=`pwd`
mkdir build && cd $_
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
cpack -G DEB
```

- check package contents

```bash
cd $ROOT_DIR/packages
dpkg-deb -R pangolin_0.8.1_amd64.deb ./package
cat package/DEBIAN/control
```

- install

```bash
sudo apt-get install -f ./pangolin_0.8.1_amd64.deb
```

## :gem: References ##
***

- [Making a deb package with CMake/CPack and hosting it in a private APT repository](https://decovar.dev/blog/2021/09/23/cmake-cpack-package-deb-apt/)
- [How do I create a ppa ](https://askubuntu.com/questions/71510/how-do-i-create-a-ppa)
