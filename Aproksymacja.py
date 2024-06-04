import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
#Kacper Chwiedor 5.04.24
#transmitancja Laplace'a
def transformata_laplasa(s, k, tau, zeta, tau_z):
    return (k * (tau_z * s + 1)) / (tau**2 * s**2 + 2 * tau * zeta * s + 1)

#funkcja odpowiedzi skokowej dla transmitancji drugiego rzędu
def odpowiedz_skokowa(t, tau, zeta):
    return 1 - np.exp(-zeta * t / tau) * (np.cos(np.sqrt(1 - zeta**2) * t / tau) + (zeta / np.sqrt(1 - zeta**2)) * np.sin(np.sqrt(1 - zeta**2) * t / tau))

#funkcja błędu - regresja nieliniowa
def funkcja_bledu(params, t, h_t):
    k, tau, zeta, tau_z = params
    predicted_h_t = k * (tau_z * odpowiedz_skokowa(t, tau, zeta) + h_t[0])
    return np.sum((predicted_h_t - h_t)**2)


data = np.loadtxt("data8.txt")
t = data[:, 0]
h_t = data[:, 1]

# początkowe wartości parametrów z przykladu
initial_guess = [1, 1, 0.5, 1]

# minimalizacja funkcji błędu przy użyciu gotowego minimize
result = minimize(funkcja_bledu, initial_guess, args=(t, h_t), method='Nelder-Mead')

print("Najlepsze parametry:")
print("k =", result.x[0])
print("tau =", result.x[1])
print("zeta =", result.x[2])
print("tau_z =", result.x[3])

plt.plot(t, h_t, 'bo', label='Dane pomiarowe')
t_theoretical = np.linspace(0, max(t), 1000)
h_theoretical = result.x[0] * (result.x[3] * odpowiedz_skokowa(t_theoretical, result.x[1], result.x[2]) + h_t[0])
plt.plot(t_theoretical, h_theoretical, 'r-', label='Teoretyczna odpowiedź skokowa')
plt.xlabel('Czas')
plt.ylabel('Odpowiedź skokowa')
plt.title('Dane pomiarowe i teoretyczna odpowiedź skokowa')
plt.legend()
plt.grid(True)
plt.show()
