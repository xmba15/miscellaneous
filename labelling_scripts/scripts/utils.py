#!/usr/bin/env python


__all__ = ["get_all_files_with_format_from_path"]


def get_all_files_with_format_from_path(dir_path: str, suffix_formats: list, concat_dir_path=True):
    import os

    def _human_sort(s):
        """Sort list the way humans do"""
        import re

        pattern = r"([0-9]+)"
        return [int(c) if c.isdigit() else c.lower() for c in re.split(pattern, s)]

    all_files = [elem for elem in os.listdir(dir_path) if elem.split(".")[-1] in suffix_formats]
    all_files.sort(key=_human_sort)
    if concat_dir_path:
        all_files = [os.path.join(dir_path, cur_file) for cur_file in all_files]

    return all_files
