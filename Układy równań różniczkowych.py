import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#190253
# Układ równań różniczkowych (Lorenz system)
def lorenz(t, xyz):
    x, y, z = xyz
    sigma = 10
    beta = 8/3
    rho = 28
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Parametry
t0 = 0
tk = 25
h = 0.03125
y0 = [5, 5, 5]

# Metoda Rungego-Kutty 4 rzędu z pakietu scipy
t_eval = np.arange(t0, tk, h)
sol = solve_ivp(lorenz, [t0, tk], y0, method='RK45', t_eval=t_eval)

# Własna implementacja metody RK4
def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 * np.array(k1))
    k3 = f(t + h/2, y + h/2 * np.array(k2))
    k4 = f(t + h, y + h * np.array(k3))
    return y + h/6 * (np.array(k1) + 2*np.array(k2) + 2*np.array(k3) + np.array(k4))

# Własne rozwiązanie układu równań różniczkowych
t_points = np.arange(t0, tk, h)
xyz = np.zeros((len(t_points), 3))
xyz[0] = y0

for i in range(1, len(t_points)):
    xyz[i] = rk4_step(lorenz, t_points[i-1], xyz[i-1], h)

# Wykresy
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

axs[0].plot(sol.t, sol.y[0], label='scipy ')
axs[0].plot(t_points, xyz[:, 0], '-', label='moja RK4', color='magenta',linewidth=0.5)
axs[0].set_title('x vs. t')
axs[0].legend()

axs[1].plot(sol.t, sol.y[1], label='scipy ')
axs[1].plot(t_points, xyz[:, 1], '-', label='moja RK4', color='magenta', linewidth=0.5)
axs[1].set_title('y vs. t')
axs[1].legend()

axs[2].plot(sol.t, sol.y[2], label='scipy ')
axs[2].plot(t_points, xyz[:, 2], '-', label='moja RK4', color='magenta', linewidth=0.5)
axs[2].set_title('z vs. t')
axs[2].legend()

plt.tight_layout()
plt.show()

# Trajektoria fazowa
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2], label='scipy ')
ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], '-', label='moja RK4', color='magenta', linewidth=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('Trajektoria fazowa')
plt.show()
