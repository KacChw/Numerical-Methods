import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, bisect
from scipy.optimize import newton

def f1(x):
    return x ** 2 + 2 * x + 0.5

def f2(x):
    return (-2 * x ** 2) / (3 - (7 * x / 3))

x_values = np.linspace(-10, 10, 400)

# ustalamy zakres x
y1_values = f1(x_values)
y2_values = f2(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, y1_values, label='y = x^2 + 2x + 0.5')
plt.plot(x_values, y2_values, label='3y = -2x^2 + 7xy')

plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.title('Rozwiązanie graficzne układu równań nieliniowych')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Znajdujemy przybliżenia
# Na podstawie wykresu widzimy, że istnieją trzy punkty przecięcia, które są przybliżonymi rozwiązaniami
# Możemy wykorzystać funkcję fsolve z pakietu scipy do znalezienia dokładniejszych wartości pierwiastków

# Definiujemy funkcję układu równań
def equations(xy):
    x, y = xy
    eq1 = x ** 2 + 2 * x + 0.5 - y
    eq2 = 3 * y + 2 * x ** 2 - 7 * x * y
    return [eq1, eq2]


# Przybliżone wartości punktów przecięcia na osi x
x_guesses = [-5, 0, 5]

# Znajdujemy rozwiązania numeryczne dla każdej przybliżonej wartości punktu przecięcia
solutions = []
for x_guess in x_guesses:
    y_guess = f1(x_guess)  # Przybliżona wartość punktu przecięcia na osi y
    x_root, y_root = fsolve(equations, [x_guess, y_guess])
    solutions.append((x_root, y_root))

# Wyświetlamy przybliżone wartości pierwiastków
print("Liczba pierwiastków:", len(solutions))
print("Przybliżone wartości pierwiastków:")
for i, (x, y) in enumerate(solutions):
    print(f"Punkt {i + 1}: x = {x}, y = {y}")

# #--------------------------------------------podpunkt drugi-----------------------------------------------------------------

def f1(x):
    return x ** 2 + 2 * x + 0.5

def f2(x, y):
    return 2 * x ** 2 + 7 * x * y / 3

# Metoda iteracyjnego podstawiania
def iterative_substitution(f, initial_guess, tolerance=1e-6, max_iterations=1000):
    def iteration_step(x):
        return f(x)

    x = initial_guess
    for _ in range(max_iterations):
        try:
            x_next = iteration_step(x)
        except OverflowError:
            return None  # If result becomes too large, return None
        if abs(x_next - x) < tolerance:
            return x_next
        x = x_next
    return None  # Jeśli nie osiągnięto zbieżności


# Metoda bisekcji
def equation_to_solve(x, y):
    return x ** 2 + 2 * x + 0.5 - f2(x, y)


# Przybliżone rozwiązanie za pomocą metody iteracyjnego podstawiania
initial_guess = 0.5  # Changed initial guess
solution_iterative = iterative_substitution(f1, initial_guess)


if solution_iterative is not None:
    y_solution_iterative = f1(solution_iterative)
    print("Rozwiązanie za pomocą metody iteracyjnego podstawiania:")
    print("x =", solution_iterative)
    print("y =", y_solution_iterative)
else:
    print("Metoda iteracyjnego podstawiania nie znalazła rozwiązania.")

# Przybliżone rozwiązanie za pomocą metody bisekcji
solution_bisection = bisect(equation_to_solve, -10, 0, args=(0,))

if solution_bisection is not None:
    y_solution_bisection = f1(solution_bisection)
    print("\nRozwiązanie za pomocą metody bisekcji:")
    print("x =", solution_bisection)
    print("y =", y_solution_bisection)
else:
    print("Metoda bisekcji nie znalazła rozwiązania.")

#------------------------------------------------podpunkt trzeci-------------------------------------------------------
from scipy.optimize import newton

# Definiujemy funkcje
def f1(x):
    return x**2 + 2*x + 0.5

def f2(x, y):
    return 2*x**2 + 7*x*y/3

# Rozwiązanie za pomocą scipy.optimize.newton
solution_newton_f1 = newton(f1, x0=0)
print("Rozwiązanie za pomocą scipy.optimize.newton dla równania f1:")
print("x =", solution_newton_f1)
print("y =", f1(solution_newton_f1))

def f2_x_derivative(x, y):
    return 4*x - (7*y)/3

def newton_raphson_method(f, f_derivative, x0, y0, tolerance=1e-6, max_iterations=1000):
    x = x0
    for _ in range(max_iterations):
        y = f(x)
        x_next = x - f(x)/f_derivative(x, y)
        if abs(x_next - x) < tolerance:
            return x_next
        x = x_next
    return None  # Jeśli nie osiągnięto zbieżności

# Rozwiązanie za pomocą zaimplementowanej metody Newtona-Raphsona
initial_guess = 0  # Zmieniamy punkt startowy
solution_newton_raphson = newton_raphson_method(f1, f2_x_derivative, initial_guess, 0, max_iterations=10000)

if solution_newton_raphson is not None:
    print("\nRozwiązanie za pomocą zaimplementowanej metody Newtona-Raphsona dla równania f1:")
    print("x =", solution_newton_raphson)
    print("y =", f1(solution_newton_raphson))
else:
    print("\nMetoda Newtona-Raphsona nie znalazła rozwiązania dla danego punktu startowego.")

