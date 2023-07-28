import os
import json
import random
import yaml
import numpy as np

MIN_FLOW_NUMBER = 0.002
MAX_FLOW_NUMBER = 0.2
MIN_FROUDE_NUMBER = 0.2
MAX_FROUDE_NUMBER = 2.0


class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)


def calculate_gas_flow_number(volume_flow_lpm, rpm, d_stirrer=0.095):
    volume_flow_m3_per_sec = volume_flow_lpm / 60 / 1000
    rps = rpm / 60
    return volume_flow_m3_per_sec / (rps * d_stirrer ** 3)


def calculate_froude_number(rpm, d_stirrer=0.095):
    rps = rpm / 60
    return rps ** 2 * d_stirrer / 9.81


def flow_regime(gas_flow_number, froude_number):
    log_froude_number = np.log10(froude_number)
    log_gas_flow_number = np.log10(gas_flow_number)

    upper_interpolation = np.interp(log_gas_flow_number, np.log10([0.002, 0.2]), np.log10([0.2, 2.0]))
    lower_interpolation = np.interp(log_gas_flow_number, np.log10([0.013, 1]), np.log10([0.02, 1.4]))

    middle_interpolation = (upper_interpolation + lower_interpolation) / 2

    if log_froude_number >= upper_interpolation:
        return 2
    elif log_froude_number <= lower_interpolation:
        return 0
    else:
        return 1


def get_files_from_directory(path: str, extension: str = ".json") -> list:
    return [file for file in os.listdir(path) if file.endswith(extension)]


def load_json_from_file(file_path: str):
    with open(file_path, "r") as file:
        return json.load(file)


def write_json_to_file(file_path: str, data: dict):
    with open(file_path, "w") as file:
        json.dump(data, file)


def label_files_in_directory(path: str) -> dict:
    label_counts = {0: 0, 1: 0, 2: 0}
    for filename in get_files_from_directory(path):
        file_path = os.path.join(path, filename)
        json_data = load_json_from_file(file_path)
        gas_flow_number = calculate_gas_flow_number(
            json_data["gas_flow_rate"]["data"]["opcua_value"]["value"],
            json_data["stirrer_rotational_speed"]["data"]["opcua_value"]["value"]
        )
        froude_number = calculate_froude_number(
            json_data["stirrer_rotational_speed"]["data"]["opcua_value"]["value"])
        label = flow_regime(gas_flow_number, froude_number)
        json_data["flow_regime"] = {"parameters": {"value": label}}
        write_json_to_file(file_path, json_data)
        label_counts[label] += 1
    return label_counts

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualize_decisionCurves():
    flow_number = np.logspace(np.log10(0.002), np.log10(0.2), 100)
    froude_number = np.logspace(np.log10(0.013), np.log10(2.0), 100)

    FR, RPM = np.meshgrid(flow_number, froude_number)

    Z = np.vectorize(flow_regime)(FR, RPM)

    cmap = ListedColormap(['blue', 'green', 'red', 'purple'])

    plt.figure(figsize=(10, 7))
    plt.scatter(FR, RPM, c=Z, cmap=cmap, alpha=0.5, s=1)
    plt.colorbar(ticks=[0, 1, 2, 3], label="Flow regime")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Flow Number')
    plt.ylabel('Froude Number')
    plt.title('Flow regime distribution')
    #plt.xlim([np.log(MIN_FLOW_NUMBER), np.log(MAX_FLOW_NUMBER)])
    #plt.ylim([np.log(MIN_FROUDE_NUMBER), np.log(MAX_FROUDE_NUMBER)])
    plt.show()
def remove_files(files: list, path: str):
    for file in files:
        os.remove(os.path.join(path, file))  # Remove json file
        png_file = file.replace(".json", "_camera_frame.png")  # Get name of associated png file
        os.remove(os.path.join(path, png_file))  # Remove png file

def choose_files(path: str) -> list:
    labeled_files = {0: [], 1: [], 2: []}
    for file_name in get_files_from_directory(path):
        file_path = os.path.join(path, file_name)
        json_data = load_json_from_file(file_path)
        label = json_data["flow_regime"]["parameters"]["value"]
        labeled_files[label].append(file_name)
    min_len = min(len(files) for files in labeled_files.values())
    chosen_files = [random.sample(files, min_len) for files in labeled_files.values()]
    return [item for sublist in chosen_files for item in sublist]  # flatten the list

def main():
    # target = ['Dispersed', 'Transition', 'Loaded', 'Flooded']
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to params.yaml
    params_file = os.path.join(current_dir, '..', '..', 'params.yaml')

    # Read the content of the YAML file
    with open(params_file, 'r') as stream:
       params = yaml.load(stream, Loader=PrettySafeLoader)
       input_output_path = params['input_output_labeling_path']

    print("Labeling files...")
    label_counts = label_files_in_directory(input_output_path)
    print(f"Class 0 = {label_counts[0]}, Class 1 = {label_counts[1]}, Class 2 = {label_counts[2]}")

    print("Balancing dataset...")
    chosen_files = choose_files(input_output_path)
    all_files = get_files_from_directory(input_output_path)
    files_to_delete = set(all_files) - set(chosen_files)
    print(f"Total files to be deleted: {len(files_to_delete)}")
    remove_files(files_to_delete, input_output_path)

    print("Counting new label distribution after deletion...")
    new_label_counts = label_files_in_directory(input_output_path)
    print(f"Class 0 = {new_label_counts[0]}, Class 1 = {new_label_counts[1]}, Class 2 = {new_label_counts[2]}")

if __name__ == '__main__':
    visualize_decisionCurves()
    main()
def run():
    main()