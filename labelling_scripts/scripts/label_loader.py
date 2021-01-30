#!/usr/bin/env python
import base64
import json
import labelme
import os


__all__ = ["DetectionLabelLoader"]


class DetectionLabelLoader:
    @staticmethod
    def load_signate(json_path: str):
        assert os.path.isfile(json_path)

        with open(json_path) as f:
            json_dict = json.load(f)
        assert len(json_dict) != 0

        labels = []
        bbox_list = []
        for (label, bboxes) in json_dict["labels"].items():
            for (xmin, ymin, xmax, ymax) in bboxes:
                labels.append(label)
                bbox_list.append([xmin, ymin, xmax, ymax])

        return labels, bbox_list

    @staticmethod
    def create_label_me_json_dict(img_path, img_name, img_height, img_width, labels, bbox_list):
        assert len(bbox_list) == len(labels)

        json_dict = {}
        json_dict["imagePath"] = img_name
        json_dict["imageHeight"] = img_height
        json_dict["imageWidth"] = img_width
        json_dict["imageData"] = base64.b64encode(labelme.LabelFile.load_image_file(img_path)).decode("utf-8")
        json_dict["shapes"] = []
        for bbox, label in zip(bbox_list, labels):
            xmin, ymin, xmax, ymax = bbox

            cur_shape = {}
            cur_shape["label"] = label
            cur_shape["shape_type"] = "rectangle"
            cur_shape["points"] = [[xmin, ymin], [xmax, ymax]]
            cur_shape["group_id"] = None
            cur_shape["flags"] = {}
            json_dict["shapes"].append(cur_shape)

        return json_dict
