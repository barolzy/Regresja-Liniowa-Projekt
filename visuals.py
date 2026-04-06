import matplotlib.pyplot as plt
import numpy as np


def plot_regression_results(Y_test, y_pred):
    # To jest Twój kod z punktu 10
    plt.scatter(Y_test, y_pred)
    plt.xlabel("Wartości rzeczywiste")
    plt.ylabel("Wartości przewidywane")
    plt.title("Regresja Liniowa - Przewidywanie kosztów ubezpieczenia")

    max_val = max(Y_test.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], color="black", linestyle="--")
    plt.show()


def calculate_r2(Y_test, y_pred):
    # To jest Twój kod z punktu 11
    ss_res = np.sum((Y_test - y_pred) ** 2)
    ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def print_theta(theta):
    # To jest Twój print z końca kodu
    print("Parametry modelu (theta):")
    print(theta)