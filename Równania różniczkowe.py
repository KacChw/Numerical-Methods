import numpy as np
import matplotlib.pyplot as plt
#190253 - Kacper Chwiedor

def funkcja(t, y):
    return y * (1 - y) * (2 * t - 3)

#  Eulera
def euler(f, t0, y0, h, kroki):
    t_values = [t0]
    y_values = [y0]
    for _ in range(kroki):
        y_new = y_values[-1] + h * f(t_values[-1], y_values[-1])
        t_values.append(t_values[-1] + h)
        y_values.append(y_new)
    return t_values, y_values

#  Heuna
def heun(f, t0, y0, h, kroki):
    t_values = [t0]
    y_values = [y0]
    for _ in range(kroki):
        y_predict = y_values[-1] + h * f(t_values[-1], y_values[-1])
        y_new = y_values[-1] + 0.5 * h * (f(t_values[-1], y_values[-1]) + f(t_values[-1] + h, y_predict))
        t_values.append(t_values[-1] + h)
        y_values.append(y_new)
    return t_values, y_values

# Metoda punktu środkowego (Runge-Kutta rzędu 2)
def midpoint(f, t0, y0, h, kroki):
    t_values = [t0]
    y_values = [y0]
    for _ in range(kroki):
        t = t_values[-1]
        y = y_values[-1]
        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        y_new = y + k2
        t_values.append(t + h)
        y_values.append(y_new)
    return t_values, y_values

#  analityczne
def analytic_solution(t, C):
    return 1 / (1 + np.exp(-(t**2 - 3*t + C)))

t0 = 0
y0 = 2
h = 0.01
kroki = 100
tk = 3

#stała całkowania C
C = np.log(3 - 3*y0) - t0**2 + 3*t0
#metoda Eulera
t_values_euler, y_values_euler = euler(funkcja, t0, y0, h, kroki)
#metoda Heuna
t_values_heun, y_values_heun = heun(funkcja, t0, y0, h, kroki)
#metodapunktu środkowego
t_values_midpoint, y_values_midpoint = midpoint(funkcja, t0, y0, h, kroki)

t_analytic = np.linspace(t0, tk, 100)
y_analytic = analytic_solution(t_analytic, C)

plt.plot(t_analytic, y_analytic, label='Rozwiązanie analityczne', color='black')
plt.plot(t_values_euler, y_values_euler, label='Metoda Eulera', linestyle='--')
plt.plot(t_values_heun, y_values_heun, label='Metoda Heuna', linestyle='-.')
plt.plot(t_values_midpoint, y_values_midpoint, label='Metoda punktu środkowego', linestyle=':')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Porównanie metod numerycznych')
plt.legend()
plt.grid(True)
plt.show()
