import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from functions import gradient_descent
from visuals import plot_regression_results, calculate_r2, print_theta


data = pd.read_csv('insurance.csv')

data['sex'] = data['sex'].map({'male': 1, 'female': 0})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})

X = data[['age', 'sex', 'bmi', 'children', 'smoker']].copy()
Y = data['charges']
# -----------------------------------
with open('parameters.json', 'r') as file:
    params = json.load(file)
    alpha = params['alpha']
    num_iters = params['num_iters']



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.60, random_state=42)

mean_values = X_train.mean()
std_values = X_train.std()
X_train = (X_train - mean_values) / std_values
X_test = (X_test - mean_values) / std_values

X_train = X_train.values
X_test = X_test.values
Y_train = Y_train.values.reshape(-1, 1)
Y_test = Y_test.values.reshape(-1, 1)

X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

theta = np.zeros((X_train.shape[1], 1))

theta, cost_history = gradient_descent(X_train, Y_train, theta, alpha, num_iters)

y_pred = np.dot(X_test, theta)


plot_regression_results(Y_test, y_pred)

r2 = calculate_r2(Y_test, y_pred)
print(f"Współczynnik R^2: {r2}")
print_theta(theta)
#komentarz do 9 zadania blblbabal