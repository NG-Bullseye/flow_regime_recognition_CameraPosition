import csv
import os
from datetime import datetime


def main(results,output_dir="/home/lwecke/PycharmProjects/flow_regime_recognition_CameraPosition/modules"):
    filename = f"results{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Full path to the file
    file_path = os.path.join(output_dir, filename)

    # Writing to csv file
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Key', 'Value'])
        for key, value in results.items():
            writer.writerow([key, value])