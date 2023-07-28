import os
import training as model_training
from modules.Utility.PrettySafeLoader import PrettySafeLoader
from modules import Utility

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
        trainer = model_training.Training(subdirectory,yaw_value+output_bulktraining_path,1,54)  # Or however you initialize your training object
        trainer.train()
        trainer.evaluate()
        trainer.report()
        # Store the validation accuracy for this yaw value
        results[yaw_value] = trainer.results[1]  # Assuming trainer.results[1] is validation accuracy

    return results


def main():
    import os
    import yaml
    from datetime import datetime

    PrettySafeLoader.add_constructor(
        u'tag:yaml.org,2002:python/tuple',
        PrettySafeLoader.construct_python_tuple)
    with open('./params.yaml', 'r') as stream:
        params = yaml.load(stream, Loader=PrettySafeLoader)

    # Get the current time
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Check if "Bulktraining" is present at the end of the path
    if params['output_training_path'].endswith('/Bulktraining'):
        # Replace the existing timestamp with the current time
        params['output_training_path'] = params['output_training_path'][:-(len('/Bulktraining'))] + f'{current_time}'
    else:
        # Append "/Bulktraining<currentTime>"
        params['output_training_path'] = os.path.join(params['output_training_path'], f'Bulktraining{current_time}')

    # Save the modified data back to the YAML file
    with open('./params.yaml', 'w') as file:
        yaml.dump(params, file)
    # Example usage:
    with open('./params.yaml', 'r') as stream:
        params = yaml.load(stream, Loader=PrettySafeLoader)
    output_bulktraining_path=    params['output_bulktraining_path']
    input_bulktraining_image_root_path = params['input_bulktraining_image_root_path']
    yaw_to_accuracy = bulktrain(input_bulktraining_image_root_path,output_bulktraining_path)
    # print yaw_to_accuracy to check the validation accuracy for each yaw value
    print(yaw_to_accuracy)
    Utility.Save_mapping_to_csv.save(output_bulktraining_path)
    # directory containing all yaw directories
    yaw_dir = "/home/lwecke/Datensätze/Datensatz_v1_50p_3reg/preprocessed_sorded_by_yaw"
    # destination directory
    dst_dir = "/home/lwecke/Datensätze/Datensatz_v1_50p_3reg/Bulktraining_Outputs/Bulktraining_2023_07_24/yaw"
    Utility.copy_trainingdata_out_of_yaw_image_folders.run(yaw_dir,dst_dir)

if __name__ == '__main__':
    main()