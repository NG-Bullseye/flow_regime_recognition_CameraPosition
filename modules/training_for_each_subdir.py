import os
from modules import model_training
from modules.Utility.PrettySafeLoader import PrettySafeLoader


def list_subdirectories(directory):
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a valid directory.")

    subdirectories = [os.path.join(directory, d) for d in os.listdir(directory) if
                      os.path.isdir(os.path.join(directory, d))]

    results = {}  # Dictionary to store validation accuracy for each yaw value

    for subdirectory in subdirectories:
        yaw_value = os.path.basename(subdirectory)
        print(f"Training for {subdirectory}...")
        # Initialize and train the model
        trainer = model_training.Training(subdirectory,1,54)  # Or however you initialize your training object
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
    bulk_training=    params['output_training_path']
    directory_path = "/home/lwecke/Datensätze/Datensatz_v1_50p_3reg/Datensatz_v1_50p_3reg/yaw"
    yaw_to_accuracy = list_subdirectories(directory_path)
    # print yaw_to_accuracy to check the validation accuracy for each yaw value
    print(yaw_to_accuracy)

if __name__ == '__main__':
    main()