#!/usr/bin/env python
import os
import tqdm


from label_loader import DetectionLabelLoader
from utils import get_all_files_with_format_from_path


def get_args():
    import argparse

    parser = argparse.ArgumentParser("Test your image or video by trained model.")
    parser.add_argument("--xmls_path", type=str, required=True)
    parser.add_argument("--images_path", type=str, required=True)
    parser.add_argument("--new_jsons_path", type=str, required=True)
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    assert os.path.isdir(args.xmls_path) and os.path.isdir(args.new_jsons_path)
    assert args.xmls_path != args.new_jsons_path

    xml_files = get_all_files_with_format_from_path(args.xmls_path, ["xml"], False)
    image_files = get_all_files_with_format_from_path(args.images_path, ["png", "jpg"], False)
    assert len(xml_files) > 0

    for xml_file, image_file in zip(tqdm.tqdm(xml_files), image_files):
        assert xml_file.split(".")[0] == image_file.split(".")[0]

        abs_image_path = os.path.join(args.images_path, image_file)

        file_name, image_height, image_width, labels, bbox_list = DetectionLabelLoader.load_voc(
            os.path.join(args.xmls_path, xml_file)
        )

        json_file = xml_file.split(".")[0] + ".json"
        DetectionLabelLoader.save_label_me_json_dict(
            abs_image_path,
            image_file,
            image_height,
            image_width,
            labels,
            bbox_list,
            os.path.join(args.new_jsons_path, json_file),
        )


if __name__ == "__main__":
    main()
