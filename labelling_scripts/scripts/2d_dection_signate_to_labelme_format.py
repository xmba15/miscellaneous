#!/usr/bin/env python
import cv2
import os
import tqdm
import json


from label_loader import DetectionLabelLoader
from utils import get_all_files_with_format_from_path


def get_args():
    import argparse

    parser = argparse.ArgumentParser("Test your image or video by trained model.")
    parser.add_argument("--jsons_path", type=str, required=True)
    parser.add_argument("--images_path", type=str, required=True)
    parser.add_argument("--new_jsons_path", type=str, required=True)
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    assert os.path.isdir(args.jsons_path) and os.path.isdir(args.new_jsons_path)
    assert args.jsons_path != args.new_jsons_path

    json_files = get_all_files_with_format_from_path(args.jsons_path, ["json"], False)
    image_files = get_all_files_with_format_from_path(args.images_path, ["png", "jpg"], False)
    assert len(json_files) > 0

    for json_file, image_file in zip(tqdm.tqdm(json_files), image_files):
        assert json_file.split(".")[0] == image_file.split(".")[0]

        labels, bbox_list = DetectionLabelLoader.load_signate(os.path.join(args.jsons_path, json_file))
        abs_image_path = os.path.join(args.images_path, image_file)
        image = cv2.imread(abs_image_path)
        assert image is not None
        img_height, img_width = image.shape[:2]
        json_dict = DetectionLabelLoader.create_label_me_json_dict(
            abs_image_path, image_file, img_height, img_width, labels, bbox_list
        )

        with open(os.path.join(args.new_jsons_path, json_file), "w") as f:
            json.dump(json_dict, f)


if __name__ == "__main__":
    main()
