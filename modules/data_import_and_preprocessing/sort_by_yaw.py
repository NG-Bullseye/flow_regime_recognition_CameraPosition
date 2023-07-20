import os
import json
import shutil
import yaml
import os
import json
import random
import numpy as np

class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)

def load_json_from_file(file_path: str):
    with open(file_path, "r") as file:
        return json.load(file)


def get_files_from_directory(path: str, extension: str = ".json") -> list:
    return [file for file in os.listdir(path) if file.endswith(extension)]


def sort_files_by_yaw(path: str):
    parent_dir = os.path.dirname(path)
    for filename in get_files_from_directory(path, extension=".json"):
        file_path = os.path.join(path, filename)
        json_data = load_json_from_file(file_path)
        yaw_value = json_data["robot_yaw"]["data"]["opcua_value"]["value"]
        dest_dir = os.path.join(parent_dir, f'yaw_{yaw_value}')

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Copy json file
        shutil.copy(file_path, dest_dir)

        # Copy corresponding png file
        png_file = filename.replace(".json", "_camera_frame.png")
        png_path = os.path.join(path, png_file)
        if os.path.exists(png_path):
            shutil.copy(png_path, dest_dir)

def main():
    with open('../../params.yaml', 'r') as stream:
        params = yaml.load(stream, Loader=PrettySafeLoader)
    path = params['dest_path_preprocessed']+"/preprocessed_images"+str(params['preprocessing']['picture_width'])

    print("Sorting files by yaw...")
    sort_files_by_yaw(path)


if __name__ == '__main__':
    main()