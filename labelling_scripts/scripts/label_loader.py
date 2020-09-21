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
    def load_label_me(json_path: str):
        assert os.path.isfile(json_path)

        json_dict = {}
        with open(json_path) as f:
            json_dict = json.load(f)
        assert len(json_dict) != 0

        file_name = json_dict["imagePath"]
        image_height = json_dict["imageHeight"]
        image_width = json_dict["imageWidth"]

        labels = []
        bbox_list = []
        for obj in json_dict["shapes"]:
            labels.append(obj["label"])
            [xmin, ymin], [xmax, ymax] = obj["points"]
            bbox_list.append([xmin, ymin, xmax, ymax])

        return file_name, image_height, image_width, labels, bbox_list

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

    @staticmethod
    def save_label_me_json_dict(img_path, img_name, img_height, img_width, labels, bbox_list, output_path):
        with open(output_path, "w") as f:
            json.dump(
                DetectionLabelLoader.create_label_me_json_dict(
                    img_path, img_name, img_height, img_width, labels, bbox_list
                ),
                f,
            )

    @staticmethod
    def load_voc(xml_path: str):
        import xml.etree.ElementTree as ET

        assert os.path.isfile(xml_path)

        with open(xml_path) as f:
            tree = ET.parse(f)
            root = tree.getroot()

        file_name = root.find("filename").text
        image_height = int(root.find("size").find("height").text)
        image_width = int(root.find("size").find("width").text)

        labels = []
        bbox_list = []

        for obj in root.iter("object"):
            labels.append(obj.find("name").text)
            xmlbox = obj.find("bndbox")
            bbox_list.append(
                [
                    float(xmlbox.find("xmin").text),
                    float(xmlbox.find("ymin").text),
                    float(xmlbox.find("xmax").text),
                    float(xmlbox.find("ymax").text),
                ]
            )

        return file_name, image_height, image_width, labels, bbox_list

    @staticmethod
    def create_voc(file_name, image_height, image_width, labels, bbox_list):
        from lxml.etree import Element, SubElement, tostring

        node_root = Element("annotation")
        node_folder = SubElement(node_root, "folder")
        node_folder.text = ""

        node_filename = SubElement(node_root, "filename")
        node_filename.text = file_name

        node_size = SubElement(node_root, "size")
        node_width = SubElement(node_size, "width")
        node_width.text = str(image_width)
        node_height = SubElement(node_size, "height")
        node_height.text = str(image_height)

        node_depth = SubElement(node_size, "depth")
        node_depth.text = ""

        node_segmented = SubElement(node_root, "depth")
        node_segmented.text = "0"

        for label, (xmin, ymin, xmax, ymax) in zip(labels, bbox_list):
            node_object = SubElement(node_root, "object")
            node_name = SubElement(node_object, "name")
            node_name.text = label

            node_occluded = SubElement(node_object, "occluded")
            node_occluded.text = "0"
            node_bndbox = SubElement(node_object, "bndbox")
            node_xmin = SubElement(node_bndbox, "xmin")
            node_xmin.text = str(xmin)
            node_ymin = SubElement(node_bndbox, "ymin")
            node_ymin.text = str(ymin)
            node_xmax = SubElement(node_bndbox, "xmax")
            node_xmax.text = str(xmax)
            node_ymax = SubElement(node_bndbox, "ymax")
            node_ymax.text = str(ymax)

        return tostring(node_root, pretty_print=True)

    @staticmethod
    def save_voc(file_name, image_height, image_width, labels, bbox_list, output_path):
        with open(output_path, "wb") as f:
            f.write(DetectionLabelLoader.create_voc(file_name, image_height, image_width, labels, bbox_list))
