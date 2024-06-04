import sympy as sp
import numpy as np

x = sp.symbols('x')
f = -0.03421*x**4 + 0.4325*x**3 - 0.4531*x**2 + 2.42*x + 5.1

# Obliczenie całki analitycznie
integral_analytical = sp.integrate(f, (x, -6, 12))

# Współczynniki wielomianu wynikowego
coefficients = [integral_analytical.coeff(x, n) for n in range(5)]

print("Całka analityczna:", integral_analytical)
print("Współczynniki wielomianu wynikowego:", coefficients)
print("\n")

def function(x):
    return -0.03421*x**4 + 0.4325*x**3 - 0.4531*x**2 + 2.42*x + 5.1

#  reguła trapezów
def trapezoidal_rule(func, a, b, n):
    h = (b - a) / n
    integral = (func(a) + func(b)) / 2
    for i in range(1, n):
        integral += func(a + i * h)
    integral *= h
    return integral

results = []
for n in range(2, 11):
    integral_numerical = trapezoidal_rule(function, -6, 12, n)
    absolute_error = np.abs(integral_analytical - integral_numerical)
    relative_error = (absolute_error / np.abs(integral_analytical)) * 100
    results.append((n, integral_numerical, absolute_error, relative_error))

for result in results:
    print("{:<10} {:.6f} {:.6f} {:.6f}%".format(result[0], result[1], result[2], result[3]))

def romberg_integration(func, a, b, epsilon=0.002):
    def richardson(r, k):
        return r[-1][-1] + (r[-1][-1] - r[-2][-1]) / (4**k - 1)

    n = 1
    h = b - a
    R = [[(h / 2) * (func(a) + func(b))]]

    while True:
        n *= 2
        h /= 2
        R.append([0.5 * R[0][0] + sum(func(a + (2*i - 1) * h) for i in range(1, n + 1)) * h])

        for m in range(1, len(R)):
            R[m].append(richardson(R, m))

        if abs(R[-1][-1] - R[-2][-1]) < epsilon:
            break

    return R[-1][-1]

integral_romberg = romberg_integration(function, -6, 12)

print("Całka metodą Romberga:", integral_romberg)
print("Błąd względny:", abs(integral_romberg - integral_analytical) / abs(integral_analytical) * 100, "%")



# Implementacja trzypunktowej kwadratury Gaussa
def gauss_quadrature_3_points(func, a, b):
    x1 = -0.7745966692
    x2 = 0
    x3 = 0.7745966692

    c1 = 0.5555555556
    c2 = 0.8888888889
    c3 = 0.5555555556

    integral = c1 * func(0.5 * (b - a) * x1 + 0.5 * (b + a)) + \
               c2 * func(0.5 * (b - a) * x2 + 0.5 * (b + a)) + \
               c3 * func(0.5 * (b - a) * x3 + 0.5 * (b + a))

    return 0.5 * (b - a) * integral


# Obliczenie całki trzypunktową kwadraturą Gaussa
integral_gauss = gauss_quadrature_3_points(function, -6, 12)

print("Całka trzypunktową kwadraturą Gaussa:", integral_gauss)
