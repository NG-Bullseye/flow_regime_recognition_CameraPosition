import csv
import os
def save(mapping,output_dir,filename):
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Full path to the file
    file_path = os.path.join(output_dir, filename)

    # Writing to csv file
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Key', 'Value'])
        for key, value in mapping.items():
            writer.writerow([key, value])