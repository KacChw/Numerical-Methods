import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, linalg

numerator_coeffs = [1, 1, 1]  # Licznik: z^2 + z + 1
denominator_coeffs = [1, -4.4, 4.65, -1.35]  # Mianownik: z^3 - 4.4*z^2 + 4.65*z - 1.35
system = signal.dlti(numerator_coeffs, denominator_coeffs)
t_impulse, response_impulse = signal.dimpulse(system, n=40)
poles = np.roots(denominator_coeffs)

print("Bieguny transmitancji:", poles)

# Stabilność układu
if all(np.real(pole) < 0 for pole in poles):
    print("Układ jest stabilny.")
else:
    print("Układ jest niestabilny.")

# Odpowiedź impulsowa
plt.stem(t_impulse, np.squeeze(response_impulse))
plt.xlabel('Czas (k)')
plt.ylabel('Odpowiedź impulsowa')
plt.title('Odpowiedź impulsowa układu')
plt.grid(True)
plt.show()

# Model stanowy układu
A, B, C, D = signal.tf2ss(numerator_coeffs, denominator_coeffs)

print("Macierz A:")
print(A)
print("Macierz B:")
print(B)
print("Macierz C:")
print(C)
print("Macierz D:")
print(D)

# odpowiedź skokowa
n_samples = 40
u = np.ones(n_samples)  # skok jednostkowy
x = np.zeros((A.shape[0], 1))  # poczatek
y = []

for _ in range(n_samples):
    y.append(np.dot(C, x)[0][0])
    x = np.dot(A, x) + np.dot(B, u[_])

plt.figure()
plt.stem(range(n_samples), y)
plt.xlabel('Czas (k)')
plt.ylabel('Odpowiedź skokowa')
plt.title('Odpowiedź skokowa układu')
plt.grid(True)
plt.show()

# LQR
c1 = 0.5
c2 = 0.1
Q = c1 * np.eye(A.shape[0])
R = c2

# Rozwiązanie równania Riccatiego
P = linalg.solve_discrete_are(A, B, Q, R)

# Obliczenie macierzy sterowania K
K = np.dot(np.dot(linalg.inv(np.dot(np.dot(B.T, P), B) + R), B.T), P)

# Modyfikacja macierzy stanu
F = K
A_nowa = A - np.dot(B, F)

# Symulacja układu z kontrolerem LQR i modyfikowaną macierzą A
x_nowa = np.zeros((A_nowa.shape[0], 1))  # Początkowy stan
y_res = []

for _ in range(n_samples):
    # Obliczenie sterowania przy użyciu sterowania LQR
    u = -np.dot(F, x_nowa)

    # Obliczenie nowego stanu układu
    x_nowa = np.dot(A_nowa, x_nowa) + np.dot(B, u)

    # Obliczenie odpowiedzi układu
    y_nowa = np.dot(C, x_nowa) + np.dot(D, u)
    y_res.append(y_nowa)

# Konwersja listy y_res na tablicę NumPy i zmiana kształtu na dwuwymiarowy
y_res_array = np.array(y_res).reshape(-1, 1)

# Wykres odpowiedzi skokowej układu z kontrolerem
plt.figure()
plt.stem(range(n_samples), y_res_array)
plt.xlabel('Czas (k)')
plt.ylabel('Odpowiedź skokowa')
plt.title('Odpowiedź skokowa układu z kontrolerem LQR')
plt.grid(True)
plt.show()

