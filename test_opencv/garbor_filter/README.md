# Method to generate multiple images to compensate the tiltness of the eyes passed through Garbor filter #

## Method Details ##

```c++
/**
 * Generate multiple images to compensate the tiltness of the eyes
 *
 * @param img: single channel image of the iris region filtered by Garbor filters
 * @param regionWidth: width of the region to extract
 * @param imageNum: number of images to be generated on each side
 * @return vector of images generated to compensate the tiltness of the eyes. Size of vector would be imageNum * 2
 */
std::vector<cv::Mat> generateAlignedImages(const cv::Mat& img,
                                           const int& regionWidth = 2,
                                           const int& imageNum = 3);
```

## How to build ##

```bash
mkdir build
cd build
cmake ../
make -j `nproc`
./garborfilters
```
