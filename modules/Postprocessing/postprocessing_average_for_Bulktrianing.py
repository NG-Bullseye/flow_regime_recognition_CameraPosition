import matplotlib.pyplot as plt
import numpy as np
import csv
import postprocessing_ref

def test(datapath):
    # Simulate what your main function might return
    # Replace this with your actual implementation
    return {
        f'yaw_{i}': 0.5 for i in range(0, 180, 10)
    }


def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Yaw", "Accuracy"])
        for key, value in data.items():
            writer.writerow([key, value])


def weighted_average_dicts(dict_list, weight_list):
    result = {}
    for d, w in zip(dict_list, weight_list):
        for key in d.keys():
            if key in result:
                result[key] += w * d[key]
            else:
                result[key] = w * d[key]
    total_weight = sum(weight_list)
    for key in result.keys():
        result[key] /= total_weight
    return result

def yawacc():
    datapaths = [
        "/home/lwecke/Datensätze/Test_datensatz_output/not_evaluated/Test_n0",
        "/home/lwecke/Datensätze/Test_datensatz_output/not_evaluated/Test_n1",
        "/home/lwecke/Datensätze/Test_datensatz_output/not_evaluated/Test_n2",
        "/home/lwecke/Datensätze/Test_datensatz_output/not_evaluated/Test_n3",
        "/home/lwecke/Datensätze/Test_datensatz_output/evaluated/n_repeats/Bulktraining_n_repeat0",
        "/home/lwecke/Datensätze/Test_datensatz_output/evaluated/n_repeats/Bulktraining_n_repeat1",
        "/home/lwecke/Datensätze/Test_datensatz_output/evaluated/n_repeats/Bulktraining_n_repeat2",
        "/home/lwecke/Datensätze/Test_datensatz_output/evaluated/n_repeats/Bulktraining_n_repeat3",
        "/home/lwecke/Datensätze/Test_datensatz_output/evaluated/n_repeats/Bulktraining_n_repeat4",
        "/home/lwecke/Datensätze/Test_datensatz_output/not_evaluated/Test_dropout"

    ]

    yaw_acc_mappings = []
    weights = []
    for datapath in datapaths:
        yaw_acc_mapping = postprocessing_ref.yawacc(datapath)
        yaw_acc_mappings.append(yaw_acc_mapping)

        # Write each yaw_acc_mapping to a CSV file
        csv_filename = f"{datapath.split('/')[-1]}.csv"
        write_to_csv(yaw_acc_mapping, csv_filename)

        # Calculate weights based on the number of entries in each yaw_acc_mapping
        weights.append(len(yaw_acc_mapping))

    weighted_result = weighted_average_dicts(yaw_acc_mappings, weights)

    print(weighted_result)
    # Convert keys to strings if they are not already
    str_keys_weighted_result = {str(key): value for key, value in weighted_result.items()}
    # Extract yaws and accuracies for plotting
    yaws = [float(key.split('_')[1]) if '_' in key else float(key) for key in str_keys_weighted_result.keys()]
    accuracies = list(str_keys_weighted_result.values())

    # Sort yaws and accuracies
    sorted_indices = np.argsort(yaws)
    sorted_yaws = np.array(yaws)[sorted_indices]
    sorted_accuracies = np.array(accuracies)[sorted_indices]

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.scatter(sorted_yaws, sorted_accuracies, color='blue')
    plt.xlabel('Yaw Angle')
    plt.ylabel('Weighted Average Accuracy')
    plt.title('Weighted Average Accuracy for Each Yaw Angle')
    plt.grid(True)
    plt.show()
    return sorted_yaws,sorted_accuracies
def yawrec():
    datapaths = [
        #"/home/lwecke/Datensätze/Test_datensatz_output/not_evaluated/Test_n0",
        #"/home/lwecke/Datensätze/Test_datensatz_output/not_evaluated/Test_n1",
        "/home/lwecke/Datensätze/Test_datensatz_output/not_evaluated/Test_n2",
        "/home/lwecke/Datensätze/Test_datensatz_output/not_evaluated/Test_n3",
        "/home/lwecke/Datensätze/Test_datensatz_output/evaluated/n_repeats/Bulktraining_n_repeat0",
        "/home/lwecke/Datensätze/Test_datensatz_output/evaluated/n_repeats/Bulktraining_n_repeat1",
        "/home/lwecke/Datensätze/Test_datensatz_output/evaluated/n_repeats/Bulktraining_n_repeat2",
        "/home/lwecke/Datensätze/Test_datensatz_output/evaluated/n_repeats/Bulktraining_n_repeat3",
        "/home/lwecke/Datensätze/Test_datensatz_output/evaluated/n_repeats/Bulktraining_n_repeat4",
        "/home/lwecke/Datensätze/Test_datensatz_output/not_evaluated/Test_dropout"
    ]

    yaw_rec_mappings = []
    weights = []
    for datapath in datapaths:
        yaw_acc_mapping, yaw_rec_mapping = postprocessing_ref.main(datapath)
        yaw_rec_mappings.append(yaw_rec_mapping)

        # Write each yaw_rec_mapping to a CSV file
        csv_filename = f"{datapath.split('/')[-1]}_rec.csv"
        write_to_csv(yaw_rec_mapping, csv_filename)

        # Calculate weights based on the number of entries in each yaw_rec_mapping
        weights.append(len(yaw_rec_mapping))

    weighted_result = weighted_average_dicts(yaw_rec_mappings, weights)

    # Convert keys to strings if they are not already
    str_keys_weighted_result = {str(key): value for key, value in weighted_result.items()}

    # Extract yaws and rec for plotting
    yaws = [float(key.split('_')[1]) if '_' in key else float(key) for key in str_keys_weighted_result.keys()]
    rec = list(str_keys_weighted_result.values())

    # Sort yaws and rec
    sorted_indices = np.argsort(yaws)
    sorted_yaws = np.array(yaws)[sorted_indices]
    sorted_rec = np.array(rec)[sorted_indices]

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.scatter(sorted_yaws, sorted_rec, color='green')
    plt.xlabel('Yaw Angle')
    plt.ylabel('Weighted Average Rec')
    plt.title('Weighted Average Rec for Each Yaw Angle')
    plt.grid(True)
    plt.show()
    return sorted_yaws, sorted_rec


if __name__ == '__main__':

    print(yawrec())

# "(array([-43.163265, -41.32653, -39.489796, -37.65306, -35.816326,
#         -33.97959, -32.142857, -30.306122, -28.469387, -26.632652,
#         -24.795918, -22.959183, -21.12245, -19.285715, -17.44898,
#         -15.612245, -13.77551, -11.938775, -10.102041, -8.265306,
#         -6.428571, -4.591837, -2.7551022, -0.9183673, 0.9183673,
#         2.7551022, 4.591837, 6.428571, 8.265306, 10.102041,
#         11.938775, 13.77551, 15.612245, 17.44898, 19.285715,
#         21.12245, 22.959183, 24.795918, 26.632652, 28.469387,
#         30.306122, 32.142857, 33.97959, 35.816326, 37.65306,
#         39.489796, 41.32653, 43.163265]), array([57.40237475, 83.96669613, 97.44749188, 111.82078625,
#                                                  105.77983, 130.96744313, 134.95452125, 112.50668,
#                                                  131.03681687, 158.58576188, 140.889825, 150.37198813,
#                                                  149.22322075, 123.14647375, 145.33259625, 153.40310575,
#                                                  110.3346175, 115.70455313, 85.71597775, 115.86194612,
#                                                  79.8357785, 59.08768025, 79.66793363, 49.61374562,
#                                                  41.27510844, -4.84240043, -0.66575887, -10.18758672,
#                                                  -29.95027181, -66.60690705, -114.52232813, -95.85175988,
#                                                  -91.78101263, -116.96814525, -126.8066155, -153.836955,
#                                                  -127.76253562, -135.48683537, -180.35181075, -113.2164235,
#                                                  -158.78573, -106.35794375, -98.4245395, -125.045426,
#                                                  -118.82992125, -150.97505613, -119.58646625, -98.82682375]))"