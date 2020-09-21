# Stereo Matching #
***

## Dataset ##
***

- Kitti Stereo 2015
```
https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip
```

*left images, right images are in training/image_2; training/image_3 directories respectively*

## How to Run ##
***

```bash
./build/stereo_vision ./data/000012_11_left.png ./data/000012_11_right.png
```

## References ##
***

- [sgm_sample](https://github.com/koichiHik/sgm_sample)
- [semi-global-matching by gishi523](https://github.com/gishi523/semi-global-matching)
