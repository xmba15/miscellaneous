# 📝 lib name
***

## :tada: TODO
***

- [x] a
- [ ] b

## 🎛  Dependencies
***
```bash
conda env create --file environment.yml
conda activate onnx
```

## :running: How to Run ##
***

- fish detection: download [fish onnx detection model](https://drive.google.com/file/d/1RQWkKxbglcrHpQ6tb5F_tYi9AuGxx0o4/view?usp=sharing)

```bash
python scripts/test_onnx_detector.py --onnx_model_path ./data/fish_yolox_l.onnx --labels "Fish" --image_path ./data/underwater.jpg
```

## :gem: References ##
***
