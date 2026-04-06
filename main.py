import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from functions import gradient_descent

# 1. Wczytanie parametrów z pliku JSON
with open('parameters.json', 'r') as file:
    params = json.load(file)
    alpha = params['alpha']
    num_iters = params['num_iters']

# 2. Wczytanie danych
data = pd.read_csv('insurance.csv')

# Mapowanie zmiennych kategorycznych na liczbowe
data['sex'] = data['sex'].map({'male': 1, 'female': 0})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})

# 3. Wybór cech (pomijamy kolumnę 'region')
X = data[['age', 'sex', 'bmi', 'children', 'smoker']].copy()
Y = data['charges']

# 4. Podział na dane treningowe (80%) i testowe (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# 5. Normalizacja danych
mean_values = X_train.mean()
std_values = X_train.std()

X_train = (X_train - mean_values) / std_values
X_test = (X_test - mean_values) / std_values

# 6. Zamiana danych na tablice NumPy i modyfikacja kształtów (reshape)
X_train = X_train.values
X_test = X_test.values

Y_train = Y_train.values.reshape(-1, 1)
Y_test = Y_test.values.reshape(-1, 1)

# Dodanie kolumny jedynek (bias - wyraz wolny theta_0)
m_train = X_train.shape[0]
X_train = np.c_[np.ones(m_train), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# 7. Inicjalizacja wektora parametrów theta
theta = np.zeros((X_train.shape[1], 1))

# 8. Trenowanie modelu
theta, cost_history = gradient_descent(X_train, Y_train, theta, alpha, num_iters)

# 9. Predykcja na danych testowych
y_pred = np.dot(X_test, theta)

# 10. Wizualizacja wyników
plt.scatter(Y_test, y_pred)
plt.xlabel("Wartości rzeczywiste")
plt.ylabel("Wartości przewidywane")
plt.title("Regresja Liniowa - Przewidywanie kosztów ubezpieczenia")

# Linia idealnego dopasowania
max_val = max(Y_test.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val], color="black", linestyle="--")
plt.show()

# 11. Sprawdzanie współczynnika R²
ss_res = np.sum((Y_test - y_pred) ** 2)
ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"Współczynnik R^2: {r2}")
print("Parametry modelu (theta):")
print(theta)