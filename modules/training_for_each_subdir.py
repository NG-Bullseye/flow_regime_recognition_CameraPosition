import os
from modules import model_training


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
    # Example usage:
    directory_path = "/home/lwecke/Datensätze/Datensatz_v1_50p_3reg/Datensatz_v1_50p_3reg/yaw"
    yaw_to_accuracy = list_subdirectories(directory_path)
    # print yaw_to_accuracy to check the validation accuracy for each yaw value
    print(yaw_to_accuracy)

if __name__ == '__main__':
    main()