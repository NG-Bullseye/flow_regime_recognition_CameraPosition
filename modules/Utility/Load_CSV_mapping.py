import csv
import os
def load(path,filename):
    # Full path to the file
    file_path = os.path.join(path, filename)
    # Reading from csv file
    results_loaded = {}
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            results_loaded[row[0]] = float(row[1])
    print("results_loaded: \n"+str(results_loaded))
    return(results_loaded)

if __name__ == '__main__':
    load()