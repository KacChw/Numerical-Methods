import numpy as np
import matplotlib.pyplot as plt

# Funkcja z laboratorium nr 1: arc sinh(x)
def function(x):
    return np.arcsinh(x)

# Procedura do obliczania przybliżenia pochodnej dla różnych wartości dx
def funDerivativeApprox(x, dx, fun):
    return (fun(x + dx) - fun(x - dx)) / (2 * dx)

# Wartość dokładna pochodnej funkcji w punkcie
#
#
# x = 0.5
exact_derivative = 1 / np.sqrt(1 + 0.5**2)

# Lista wartości dx, zaczynamy od 0.4 i dzielimy przez 5 w każdym kroku 20 razy
dx_values = [0.4 * (0.2 ** i) for i in range(20)]

# Lista do przechowywania wartości błędów dla każdego dx
absolute_errors = []

# Obliczanie błędu bezwzględnego dla każdej wartości dx
for dx in dx_values:
    approx_derivative = funDerivativeApprox(0.5, dx, function)
    absolute_errors.append(abs(exact_derivative - approx_derivative))

# Znalezienie minimalnego błędu i odpowiadającej mu wartości dx
min_error_index = np.argmin(absolute_errors)
optimal_dx = dx_values[min_error_index]

# Przedstawienie wyników w postaci tabeli
print("\ndx\t\tPrzybliżenie\tBłąd bezwzględny")
for i in range(len(dx_values)):
    print("{:.15f}\t{:.15f}\t{:.15f}".format(dx_values[i], funDerivativeApprox(0.5, dx_values[i], function), absolute_errors[i]))

# Narysowanie wykresu błędu bezwzględnego w zależności od dx
plt.figure(figsize=(10, 6))
plt.plot(dx_values, absolute_errors, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.title('Błąd bezwzględny przybliżenia pochodnej')
plt.xlabel('Wartość dx (log scale)')
plt.ylabel('Błąd bezwzględny (log scale)')
plt.grid(True)
plt.show()

# Obliczenie przybliżenia pochodnej dla optymalnej wartości dx w przedziale od 0 do 1
x_values = np.linspace(0, 1, 101)
approx_derivatives = [funDerivativeApprox(x, optimal_dx, function) for x in x_values]

# Narysowanie wykresu przybliżenia pochodnej
plt.plot(x_values, approx_derivatives)
plt.title('Przybliżenie pochodnej dla optymalnej wartości dx')
plt.xlabel('x')
plt.ylabel('Przybliżenie pochodnej')
plt.grid(True)
plt.show()
