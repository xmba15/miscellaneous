## 🎛  Dependencies ##
***

- [ceres](https://github.com/ceres-solver/ceres-solver)

## :running: How to Run ##
***

- direct pose estimation app

```bash
./build/direct_pose_estimation_app ./data/left.png ./data/000001.png ./data/disparity.png
```

the result should be something like

```text
T21=
   0.999991  0.00263142  0.00319878  0.00873198
   -0.00263851    0.999994  0.00221518   0.0012297
   -0.00319293  -0.0022236    0.999992   -0.733238
             0           0           0           1
```

- direct pose estimation python script

```bash
python scripts/test_direct_pose_estimation.py --first_image ./data/left.png --second_image ./data/000001.png --disparity_image ./data/disparity.png
```

## :gem: References ##
***

- [numba types](https://numba.pydata.org/numba-doc/dev/reference/types.html)
