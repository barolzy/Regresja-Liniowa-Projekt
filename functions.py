import numpy as np


# Funkcja obliczająca przewidywania modelu (Regresja liniowa)
def predict(X, theta):
    # Wykorzystujemy funkcję dot z biblioteki numpy do mnożenia macierzy
    return np.dot(X, theta)


# Funkcja kosztu (Mean Squared Error)
def compute_cost(X, y, theta):
    m = X.shape[0]
    h = predict(X, theta)
    return (1 / (2 * m)) * np.sum((h - y) ** 2)


# Implementacja algorytmu Gradient Descent
def gradient_descent(X, y, theta, alpha, num_iters):
    m = X.shape[0]
    J_history = []
    X_T = np.transpose(X)

    for i in range(num_iters):
        h = predict(X, theta)
        # Aktualizacja parametrów ze wzoru na gradient
        theta = theta - alpha * (1 / m) * np.dot(X_T, (h - y))
        # Zapisujemy wynik funkcji kosztu
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history