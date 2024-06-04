#dane/lab_8_dane/data8.txt
#Chwiedor Kacper 190253
import numpy as np
from scipy.interpolate import CubicSpline

def divided_diff(x, y):
    n = len(x)
    F = np.zeros((n, n))
    F[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            F[i][j] = (F[i + 1][j - 1] - F[i][j - 1]) / (x[i + j] - x[i])

    return F[0]

def newton_interpolation(x, y, xi):
    n = len(x)
    F = divided_diff(x, y)
    P = F[0]
    for i in range(1, n):
        term = F[i]
        for j in range(i):
            term *= (xi - x[j])
        P += term
    return P

def cubic_spline_interpolation(x, y):
    n = len(x)
    h = np.diff(x)
    b = np.zeros(n)
    d = np.zeros(n)
    u = np.zeros(n)
    v = np.zeros(n)
    for i in range(1, n - 1):
        b[i] = (6 / (h[i] + h[i - 1])) * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    for i in range(1, n - 1):
        u[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * v[i - 1]
        v[i] = h[i] / u[i]
        d[i] = (b[i] - h[i - 1] * d[i - 1]) / u[i]
    for i in range(n - 2, 0, -1):
        d[i] = d[i] - v[i] * d[i + 1]
    return b, d

def evaluate_cubic_spline(x, y, b, d, xi):
    n = len(x)
    for i in range(1, n):
        if x[i - 1] <= xi <= x[i]:
            dx = x[i] - x[i - 1]
            dy = y[i] - y[i - 1]
            hi = dx
            ci = dy / dx
            ai = y[i - 1]
            bi = (dy - dx * (2 * d[i - 1] + d[i])) / dx ** 2
            zi = (xi - x[i - 1]) / dx
            yi = ai + bi * zi + ci * zi ** 2 + d[i - 1] * zi ** 3
            return yi

with open('dane/lab_8_dane/data8.txt', 'r') as file:
    data = file.readlines()

points = [tuple(map(float, line.strip().split())) for line in data]

x_values = [point[0] for point in points]
y_values = [point[1] for point in points]

x_interpolated = np.linspace(min(x_values), max(x_values), 100)
y_interpolated_newton = [newton_interpolation(x_values, y_values, xi) for xi in x_interpolated]

# Dokonujemy interpolacji za pomocą funkcji sklejanych trzeciego rzędu (cubic spline)
cubic_spline = CubicSpline(x_values, y_values)
y_interpolated_cubic = cubic_spline(x_interpolated)

import matplotlib.pyplot as plt

plt.plot(x_values, y_values, 'ro', label='Punkty trajektorii')
plt.plot(x_interpolated, y_interpolated_newton, label='Interpolacja Newtona')
plt.plot(x_interpolated, y_interpolated_cubic, label='Funkcja sklejana trzeciego rzędu')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolacja trajektorii')
plt.legend()
plt.grid(True)
plt.show()

points = [tuple(map(float, line.strip().split())) for line in data]

x_values = np.array([point[0] for point in points])
y_values = np.array([point[1] for point in points])

# Dokonujemy interpolacji funkcją sklejaną trzeciego rzędu
b, d = cubic_spline_interpolation(x_values, y_values)

x_interpolated = np.linspace(min(x_values), max(x_values), 100)
y_interpolated_cubic_spline = [evaluate_cubic_spline(x_values, y_values, b, d, xi) for xi in x_interpolated]

import matplotlib.pyplot as plt

plt.plot(x_values, y_values, 'ro', label='Pomiary trajektorii')
plt.plot(x_interpolated, y_interpolated_cubic_spline, label='Funkcja sklejana trzeciego rzędu')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolacja trajektorii za pomocą funkcji sklejania trzeciego rzędu jak na wykładzie')
plt.legend()
plt.grid(True)
plt.show()

