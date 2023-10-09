import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, kendalltau, spearmanr
import postprocessing_average_for_Bulktrianing


class PolynomialPredictor:
    def __init__(self, csv_file):
        with open(csv_file, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            coeff = [float(x) for x in next(csv_reader)]
            self.intercept = coeff[0]
            self.coefficients = np.array(coeff[1:])

    def fn(self, x):
        x_poly = np.array([x ** i for i in range(len(self.coefficients))])
        return self.intercept + np.dot(self.coefficients, x_poly)

def fit_and_store_polynomial_acc(x, y, degree=11):
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    with open('fittedPolynomial_acc.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([model.intercept_[0]] + list(model.coef_[0]))
def fit_and_store_polynomial_rec(x, y, degree=11):
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    with open('fittedPolynomial_rec.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([model.intercept_[0]] + list(model.coef_[0]))


def plot_from_csv(x_values, csv_file):
    predictor = PolynomialPredictor(csv_file)
    y_values = [predictor.fn(x) for x in x_values]
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, y_values, color='green', label='Polynomial from CSV')
    plt.xlabel('X-values')
    plt.ylabel('Y-values (fn(x))')
    plt.title('Polynomial Plot from CSV')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_and_store_acc():
    sorted_yaws, sorted_accuracies = postprocessing_average_for_Bulktrianing.yawacc()
    df = pd.DataFrame({'Sorted Yaws': sorted_yaws, 'Sorted Accuracies': sorted_accuracies})
    df.to_csv('postprocessing_average_acc_for_Bulktrianing.csv', index=False)

def save_and_store_rec():#this function was created to store the results from terminal into a csv bacause they where printed only.
    import csv
    import numpy as np

    yaw_array = np.array([
        -43.163265, -41.32653, -39.489796, -37.65306, -35.816326, -33.97959,
        -32.142857, -30.306122, -28.469387, -26.632652, -24.795918, -22.959183,
        -21.12245, -19.285715, -17.44898, -15.612245, -13.77551, -11.938775,
        -10.102041, -8.265306, -6.428571, -4.591837, -2.7551022, -0.9183673,
        0.9183673, 2.7551022, 4.591837, 6.428571, 8.265306, 10.102041, 11.938775,
        13.77551, 15.612245, 17.44898, 19.285715, 21.12245, 22.959183, 24.795918,
        26.632652, 28.469387, 30.306122, 32.142857, 33.97959, 35.816326, 37.65306,
        39.489796, 41.32653, 43.163265
    ])

    acc_array = np.array([
        57.40237475, 83.96669613, 97.44749188, 111.82078625, 105.77983, 130.96744313,
        134.95452125, 112.50668, 131.03681687, 158.58576188, 140.889825, 150.37198813,
        149.22322075, 123.14647375, 145.33259625, 153.40310575, 110.3346175, 115.70455313,
        85.71597775, 115.86194612, 79.8357785, 59.08768025, 79.66793363, 49.61374562,
        41.27510844, -4.84240043, -0.66575887, -10.18758672, -29.95027181, -66.60690705,
        -114.52232813, -95.85175988, -91.78101263, -116.96814525, -126.8066155, -153.836955,
        -127.76253562, -135.48683537, -180.35181075, -113.2164235, -158.78573, -106.35794375,
        -98.4245395, -125.045426, -118.82992125, -150.97505613, -119.58646625, -98.82682375
    ])

    # Create a CSV file to store yaw and acc data
    with open('postprocessing_average_rec_for_Bulktrianing.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['yaw', 'rec_scalar'])  # Writing header

        for yaw, acc in zip(yaw_array, acc_array):
            csv_writer.writerow([yaw, acc])
def load_and_analyse_acc():
    loaded_df = pd.read_csv('postprocessing_average_acc_for_Bulktrianing.csv')
    yaws = loaded_df['Sorted Yaws'].to_numpy().reshape(-1, 1)
    accuracies = loaded_df['Sorted Accuracies'].to_numpy().reshape(-1, 1)
    degree = 11

    fit_and_store_polynomial_acc(yaws, accuracies,degree)

    correlation, _ = pearsonr(yaws.flatten(), accuracies.flatten())

    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(yaws)
    model = LinearRegression()
    model.fit(x_poly, accuracies)
    y_poly_pred = model.predict(x_poly)
    r2 = r2_score(accuracies, y_poly_pred)

    print(f'Polynomial Regression R2 score: {r2:.4f}')

    plt.figure(figsize=(12, 6))
    plt.scatter(yaws, accuracies, color='blue', label='Raw Data')
    plt.plot(yaws, y_poly_pred, color='red', label=f'Polynomial (degree={degree}, R^2={r2:.4f})')

    for i, (yaw, accuracy) in enumerate(zip(yaws, accuracies)):
        plt.text(yaw, accuracy, f"{accuracy[0]:.4f}")

    tau, p_value_tau = kendalltau(yaws, accuracies)
    coef, p_value_spear = spearmanr(yaws, accuracies)

    plt.xlabel('Yaw Angle')
    plt.ylabel('Weighted Average Accuracy')
    #plt.title(f'Weighted Average Accuracy vs Yaw Angle\nPearson\'s Correlation: {correlation:.4f}\nPolynomial Regression R2: {r2:.4f}')
    plt.legend()
    plt.grid(True)
    plt.show()

    x_values_test = np.linspace(min(yaws), max(yaws), 400)
    plot_from_csv(x_values_test, f'fittedPolynomial_acc.csv')
def load_and_analyse_rec():
    loaded_df = pd.read_csv('postprocessing_average_rec_for_Bulktrianing.csv')
    yaws = loaded_df['yaw'].to_numpy().reshape(-1, 1)
    accuracies = loaded_df['rec_scalar'].to_numpy().reshape(-1, 1)
    degree = 4

    fit_and_store_polynomial_rec(yaws, accuracies,degree)

    correlation, _ = pearsonr(yaws.flatten(), accuracies.flatten())

    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(yaws)
    model = LinearRegression()
    model.fit(x_poly, accuracies)
    y_poly_pred = model.predict(x_poly)
    r2 = r2_score(accuracies, y_poly_pred)

    print(f'Polynomial Regression R2 score: {r2:.4f}')

    plt.figure(figsize=(12, 6))
    plt.scatter(yaws, accuracies, color='blue', label='Raw Data')
    plt.plot(yaws, y_poly_pred, color='red', label=f'Regression curve')

    for i, (yaw, accuracy) in enumerate(zip(yaws, accuracies)):
        plt.text(yaw, accuracy, f"{accuracy[0]:.4f}")

    tau, p_value_tau = kendalltau(yaws, accuracies)
    coef, p_value_spear = spearmanr(yaws, accuracies)


    plt.xlabel('Yaw Angle')
    plt.ylabel('Weighted Average GradCAM++ recomandation scalar')
    #plt.title(f'Weighted Average Accuracy vs Yaw Angle\nPearson\'s Correlation: {correlation:.4f}\nPolynomial Regression R2: {r2:.4f}')
    plt.legend()
    plt.grid(True)
    plt.show()

    x_values_test = np.linspace(min(yaws), max(yaws), 10000)
    plot_from_csv(x_values_test, f'fittedPolynomial_rec.csv')

#calculate_and_store_acc()
#save_and_store_rec()
load_and_analyse_acc()
load_and_analyse_rec()
