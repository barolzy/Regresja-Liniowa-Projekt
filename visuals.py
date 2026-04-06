import matplotlib.pyplot as plt

def plot_costs(cost_history):
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, color='blue')
    plt.title("Historia kosztu (Błąd średniokwadratowy)")
    plt.xlabel("Liczba iteracji")
    plt.ylabel("Koszt")
    plt.grid(True)
    plt.show()