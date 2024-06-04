import numpy as np
import matplotlib.pyplot as plt

# Wczytanie danych z pliku
data = np.loadtxt('lab_7_dane/lab_7_dane/data7.txt')

# Liczba kroków czasowych
num_steps = len(data)

# Inicjalizacja macierzy A, C, G, H, Q, R
A = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

C = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

G = np.array([[0, 0],
              [0, 0],
              [1, 0],
              [0, 1]])

H = np.eye(4)  # Macierz jednostkowa 4x4

# Macierze kowariancji szumu procesowego i pomiarowego
Q = np.array([[0.25, 0],
              [0, 0.25]])

R = np.array([[2, 0],
              [0, 2]])

# Dane początkowe
s_0 = np.array([data[0][0], data[0][1], 0, 0])  # px, py, vx=0, vy=0


P_0 = 5 * np.eye(4)  # Inicjalizacja macierzy kowariancji stanu

# Inicjalizacja estymaty
s_hat = s_0
P = P_0

# Lista do przechowywania estymat stanu w kolejnych krokach czasowych
estimated_states = []

# Implementacja filtru Kalmana - równania aktualizacji czasu
for i in range(1, num_steps):
    # Predykcja stanu - Równania aktualizacji czasu

    s_hat_minus = np.dot(A, s_hat)
    P_minus = np.dot(np.dot(A, P), A.T) + np.dot(np.dot(G, Q), G.T)

    # Aktualizacja stanu na podstawie pomiaru - Równania aktualizacji pomiarów
    K = np.dot(np.dot(P_minus, C.T), np.linalg.inv(np.dot(np.dot(C, P_minus), C.T) + R))
    s_hat = s_hat_minus + np.dot(K, data[i] - np.dot(C, s_hat_minus))
    P = np.dot((np.eye(4) - np.dot(K, C)), P_minus)

    # Dodanie estymaty stanu do listy
    estimated_states.append(s_hat)
# Konwersja listy estymat stanu na tablicę numpy
estimated_states = np.array(estimated_states)
# Liczba kroków do przewidzenia
num_predictions = 5

# Przewidywanie stanu po 5 sekundach
predicted_states = []
for _ in range(num_predictions):
    # Predykcja stanu
    s_hat_minus = np.dot(A, s_hat)
    P_minus = np.dot(np.dot(A, P), A.T) + np.dot(np.dot(G, Q), G.T)

    # Aktualizacja estymaty stanu
    s_hat = s_hat_minus
    P = P_minus

    # Dodanie przewidywanej estymaty stanu do listy
    predicted_states.append(s_hat)

# Przygotowanie danych do wykresu
estimated_states = np.array(estimated_states)
predicted_states = np.array(predicted_states)

# Wykres
plt.figure(figsize=(10, 6))
plt.plot(data[:, 0], data[:, 1], 'x', label='Pomiary trajektorii')  # pomiary trajektorii
plt.plot(estimated_states[:, 0], estimated_states[:, 1], label='Wyznaczona trajektoria', linestyle='-', marker='o')  # wyznaczona trajektoria
plt.plot(predicted_states[:, 0], predicted_states[:, 1], label='Przewidywana trajektoria', linestyle='--', marker='o')  # przewidywana trajektoria
plt.xlabel('px')
plt.ylabel('py')
plt.title('Trajektoria samolotu')
plt.legend()
plt.grid(True)
plt.show()

# Wyświetlenie estymat stanow
for i, state in enumerate(estimated_states):
    print("Krok czasowy {}: px={} py={} vx={} vy={}".format(i+1, *state))
for j, state_pred in enumerate(predicted_states):
    print("Przewidywane położenie samolotu za 5 sekund: px={} py={} vx={} vy={}".format(*state_pred))
