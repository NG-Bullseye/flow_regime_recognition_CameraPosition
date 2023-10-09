import os
from collections import defaultdict

import training as model_training
from modules.Utility.PrettySafeLoader import PrettySafeLoader
from modules import Utility
import os
import yaml
from datetime import datetime

def bulktrain(input_bulktraining_image_root_path,output_bulktraining_path):
    if not os.path.isdir(input_bulktraining_image_root_path):
        raise ValueError(f"{input_bulktraining_image_root_path} is not a valid directory.")

    subdirectories = [os.path.join(input_bulktraining_image_root_path, d) for d in os.listdir(input_bulktraining_image_root_path) if
                      os.path.isdir(os.path.join(input_bulktraining_image_root_path, d))]

    results = {}  # Dictionary to store validation accuracy for each yaw value
    for subdirectory in subdirectories:
        yaw_value = os.path.basename(subdirectory)
        print(f"Training for {subdirectory}...")
        # Initialize and train the model
        outputpath=output_bulktraining_path + "/" + yaw_value
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        os.chdir(outputpath)
        trainer = model_training.Training(subdirectory,outputpath)  # Or however you initialize your training object
        trainer.train()
        trainer.evaluate()
        trainer.report()
        # Store the validation accuracy for this yaw value
        results[yaw_value] = trainer.results[1]  # Assuming trainer.results[1] is validation accuracy
    return results

yaw_to_accuracies = defaultdict(list)


def main(n_repeats):
    PrettySafeLoader.add_constructor(
        u'tag:yaml.org,2002:python/tuple',
        PrettySafeLoader.construct_python_tuple)
    with open('../../params.yaml', 'r') as stream:
        params = yaml.load(stream, Loader=PrettySafeLoader)

    # Store the original output path
    original_output_path = params['output_bulktraining_path']

    for i in range(n_repeats):
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        unique_folder = os.path.join(original_output_path, f'Bulktraining_n_repeat{i}_{current_time}')

        if not os.path.exists(unique_folder):
            os.makedirs(unique_folder)

        params['output_bulktraining_path'] = unique_folder

        yaw_to_accuracy = bulktrain(params['input_bulktraining_image_root_path'], unique_folder)

        # Accumulate the accuracy values for each yaw
        for yaw, accuracy in yaw_to_accuracy.items():
            yaw_to_accuracies[yaw].append(accuracy)

    # Print out the dict containing lists of accuracy values for each yaw
    print("Accumulated accuracy values for each yaw:", yaw_to_accuracies)

    # Compute and store the arithmetic average of the accuracy values for each yaw
    yaw_to_avg_accuracy = {}
    for yaw, accuracies in yaw_to_accuracies.items():
        avg_accuracy = sum(accuracies) / len(accuracies)
        yaw_to_avg_accuracy[yaw] = avg_accuracy

    # Print out the dict containing the average accuracy values for each yaw
    print("Average accuracy values for each yaw:", yaw_to_avg_accuracy)



if __name__ == '__main__':
    main(n_repeats=5)