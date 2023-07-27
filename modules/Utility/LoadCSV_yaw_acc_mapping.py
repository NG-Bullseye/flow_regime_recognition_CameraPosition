import csv
import os
def main():

    output_dir = "/home/lwecke/PycharmProjects/flow_regime_recognition_CameraPosition/modules"  # Replace with your directory
    filename = "0123results.csv"

    # Full path to the file
    file_path = os.path.join(output_dir, filename)

    # Reading from csv file
    results_loaded = {}
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            results_loaded[row[0]] = float(row[1])

    print(results_loaded)

if __name__ == '__main__':
    main()