import numpy as np
import matplotlib.pyplot as plt
import math

a = -1

"""Szereg w zadaniu jest rozbiezny, zadanie wymaga innych wartosci x1 i x2"""
def funSeriesExpansion(n, x):
    """
    Oblicza wartość n-tego rozwinięcia w szereg potęgowy funkcji arcsinh(x).

    Parameters:
        n (int): Liczba określająca stopień rozwinięcia.
        x (float): Wartość dla której ma zostać obliczone rozwinięcie.

    Returns:
        float: Wartość n-tego rozwinięcia funkcji arcsinh(x) w szereg potęgowy.
    """
    # Tablice liczników i mianowników dla parzystych i nieparzystych liczb
    numerators_even = [(-1) ** i * x ** (2 * i + 1) for i in range(n)]
    denominators_even = [(2 * i + 1) for i in range(n)]
    numerators_odd = [(-1) ** i * x ** (2 * i + 1) for i in range(1, n + 1)]
    denominators_odd = [(2 * i + 1) for i in range(1, n + 1)]

    # Obliczenie sumy szeregu
    sum_even = sum(numerators_even[i] / denominators_even[i] for i in range(n))
    sum_odd = sum(numerators_odd[i] / denominators_odd[i] for i in range(n))

    return sum_even + sum_odd


# Pierwszy wykres: Rozwinięcie funkcji arcsinh(x) w szereg potęgowy dla n=3
x_values = np.linspace(2, 10, 100)
n = 3
y_values = [funSeriesExpansion(n, x) for x in x_values]

plt.figure(1)
plt.plot(x_values, y_values, label=f'n={n}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Rozwinięcie funkcji arcsinh(x) w szereg potęgowy dla n=3')
plt.legend()
plt.grid(True)

# Drugi wykres: Rozwinięcia funkcji arcsinh(x) w szereg potęgowy dla różnych n
plt.figure(2)
selected_expansions = [0, 2, 7]
for n in selected_expansions:
    y_values = [funSeriesExpansion(n, x) for x in x_values]
    plt.plot(x_values, y_values, label=f'n={n}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Rozwinięcia funkcji arcsinh(x) w szereg potęgowy dla różnych n')
plt.legend()
plt.grid(True)

plt.show()



# Tabela błędów
x = 0.4
n_values = list(range(11))

print("n | Wartość rozwinięcia | Błąd bezwzględny | Błąd względny (%)")
print("-" * 57)
for n in n_values:
    # Obliczanie wartości szeregu dla danej wartości x i n
    series_value = funSeriesExpansion(n, x)
    # Obliczanie wartości rzeczywistej funkcji arcsinh(x)
    actual_value = np.arcsinh(x)
    # Obliczanie błędów
    absolute_error = abs(actual_value - series_value)
    relative_error = (absolute_error / actual_value) * 100 if actual_value != 0 else 0
    # Wyświetlanie wyników w tabeli
    print(f"{n} | {series_value:.10f} | {absolute_error:.10f} | {relative_error:.10f}%")
