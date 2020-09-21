#!/usr/bin/env python
import os
import tqdm


from label_loader import DetectionLabelLoader
from utils import get_all_files_with_format_from_path


def get_args():
    import argparse

    parser = argparse.ArgumentParser("Test your image or video by trained model.")
    parser.add_argument("--jsons_path", type=str, required=True)
    parser.add_argument("--new_xmls_path", type=str, required=True)
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    assert os.path.isdir(args.jsons_path) and os.path.isdir(args.new_xmls_path)
    assert args.jsons_path != args.new_xmls_path

    json_files = get_all_files_with_format_from_path(args.jsons_path, ["json"], False)
    assert len(json_files) > 0

    for json_file in tqdm.tqdm(json_files):
        file_name, image_height, image_width, labels, bbox_list = DetectionLabelLoader.load_label_me(
            os.path.join(args.jsons_path, json_file)
        )

        xml_file = json_file.split(".")[0] + ".xml"
        DetectionLabelLoader.save_voc(
            file_name,
            image_height,
            image_width,
            labels,
            bbox_list,
            os.path.join(args.new_xmls_path, xml_file),
        )


if __name__ == "__main__":
    main()
