import numpy as np
import matplotlib.pyplot as plt

# Parámetros
L = 1.0  # Longitud del dominio
T = 1.0  # Tiempo total
D = 2.0  # Coeficiente de difusión
Nx = 100  # Número de puntos de discretización espacial
Nt = 1000  # Número de puntos de discretización temporal

# Paso espacial y temporal
dx = L / (Nx - 1)
dt = T / (Nt - 1)

# Discretización del espacio y el tiempo
x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

# Inicialización de la solución u
u = np.zeros((Nx, Nt))

# Condición inicial
u[:, 0] = np.sin(np.pi * x) + 0.1 * np.sin(10 * np.pi * x)

# Precalcular constante
alpha = D * dt / (dx ** 2)

# Método de diferencias finitas (vectorizado)
for n in range(Nt - 1):
    u[1:-1, n + 1] = u[1:-1, n] + alpha * (u[2:, n] - 2 * u[1:-1, n] + u[:-2, n])
    
# Gráfica de la evolución de u para 10 valores distintos del tiempo
plt.figure(figsize=(10, 6))
for i in range(0, Nt, Nt // 9):
    plt.plot(x, u[:, i], label=f"t = {t[i]:.2f}")
plt.title('Evolución de u(x, t)')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True)
plt.show()

# Calcular la solución analítica en cada paso de tiempo
analytical_solution = np.sin(np.pi * x)[:, np.newaxis] * np.exp(-D * (np.pi ** 2) * t) + 0.1 * np.sin(10 * np.pi * x)[:, np.newaxis] * np.exp(-D * (10 * np.pi) ** 2 * t)

# Comparar la solución numérica con la solución analítica
plt.figure(figsize=(10, 6))
for i in range(0, Nt, Nt // 9):
    plt.plot(x, u[:, i], label=f"Solución numérica, t = {t[i]:.2f}")
    plt.plot(x, analytical_solution[:, i], '--', label=f"Solución analítica, t = {t[i]:.2f}")
plt.title('Comparación entre solución numérica y analítica')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True)
plt.show()