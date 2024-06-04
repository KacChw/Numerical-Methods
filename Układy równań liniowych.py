import numpy as np
'''
------------------------------------------pkt 1--------------------------------------------
Q_a = 200  # m^3/h
ca = 2  # mg/m^3
Ws = 1500  # mg/h
Q_b = 300  # m^3/h
cb = 2  # mg/m^3
E_12 = 25  # m^3/h
E_23 = 50  # m^3/h
E_35 = 25  # m^3/h
E_34 = 50  # m^3/h
Q_c = 150  # m^3/h
Q_d = 350  # m^3/h
Wg = 2500  # mg/h

# Macierz współczynników A
A = np.array([[-E_12, E_12, 0, 0, 0],
              [E_12, -(E_12 + E_23), E_23, 0, 0],
              [0, E_23, -(E_23 + E_34 + E_35), E_34, E_35],
              [0, 0, E_34, -(Q_c + E_34), 0],
              [0, 0, E_35, 0, -(Q_d + E_35)]])

# Wektor wyrazów wolnych (wektor wejść u)
b = np.array([-Ws - Q_a * ca,
              -Q_b * cb,
              0,
              0,
              -Wg])

# Rozwiąż układ równań liniowych
c = np.linalg.solve(A, b)

print("Rozwiązanie (wektor c - stężenia CO):", c)
'''
'''
import numpy as np
#------------------------------------------pkt 2--------------------------------------------

Q_a = 200  # m^3/h
ca = 2  # mg/m^3
Ws_new = 800  # mg/h - nowa wartość Ws
Q_b = 300  # m^3/h
cb = 2  # mg/m^3
E_12 = 25  # m^3/h
E_23 = 50  # m^3/h
E_35 = 25  # m^3/h
E_34 = 50  # m^3/h
Q_c = 150  # m^3/h
Q_d = 350  # m^3/h
Wg_new = 1200  # mg/h - nowa wartość Wg

# Macierz współczynników A
A = np.array([[-E_12, E_12, 0, 0, 0],
              [E_12, -(E_12 + E_23), E_23, 0, 0],
              [0, E_23, -(E_23 + E_34 + E_35), E_34, E_35],
              [0, 0, E_34, -(Q_c + E_34), 0],
              [0, 0, E_35, 0, -(Q_d + E_35)]])

# Nowy wektor wyrazów wolnych
b_new = np.array([-Ws_new - Q_a * ca,
                  -Q_b * cb,
                  0,
                  0,
                  -Wg_new])

# Rozwiąż układ równań liniowych
c_new = np.linalg.solve(A, b_new)

print("Nowe stężenia CO (wektor c) dla Ws = 800 mg/h i Wg = 1200 mg/h:", c_new)
'''
import numpy as np
from scipy.linalg import lu

# ------------------------------------------------pkt 3 --------------------------------------------
Q_a = 200  # m^3/h
ca = 2  # mg/m^3
Ws = 1500  # mg/h
Q_b = 300  # m^3/h
cb = 2  # mg/m^3
E_12 = 25  # m^3/h
E_23 = 50  # m^3/h
E_35 = 25  # m^3/h
E_34 = 50  # m^3/h
Q_c = 150  # m^3/h
Q_d = 350  # m^3/h
Wg = 2500  # mg/h

# Macierz współczynników A
A = np.array([[-E_12, E_12, 0, 0, 0],
              [E_12, -(E_12 + E_23), E_23, 0, 0],
              [0, E_23, -(E_23 + E_34 + E_35), E_34, E_35],
              [0, 0, E_34, -(Q_c + E_34), 0],
              [0, 0, E_35, 0, -(Q_d + E_35)]])

# Znajdź rozkład LU macierzy A
P, L, U = lu(A)

# Oblicz odwrotności macierzy L i U
L_inv = np.linalg.inv(L)
U_inv = np.linalg.inv(U)

# Oblicz macierz odwrotną A_inv
A_inv = np.dot(U_inv, L_inv)

print("Macierz odwrotna A_inv:")
print(A_inv)
