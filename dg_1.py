import numpy as np
import matplotlib.pyplot as plt

# Función de pérdida
def loss_function(x, y):
    return 10 - np.exp(- (x**2 + 3*y**2))

# Gradiente de la función de pérdida
def gradient(x, y):
    grad_x = 2 * x * np.exp(- (x**2 + 3*y**2))
    grad_y = 6 * y * np.exp(- (x**2 + 3*y**2))
    return grad_x, grad_y

# Parámetros del descenso de gradiente
learning_rate = 0.1
num_iterations = 10000
x_initial, y_initial = 1, 1

# Listas para almacenar los valores
x_values = [x_initial]
y_values = [y_initial]
loss_values = [loss_function(x_initial, y_initial)]

# Descenso de gradiente
for i in range(num_iterations):
    gradient_x, gradient_y = gradient(x_initial, y_initial)  # Calcula el gradiente en (x, y)
    x_update = x_initial - learning_rate * gradient_x
    y_update = y_initial - learning_rate * gradient_y
    
    x_values.append(x_update)
    y_values.append(y_update)
    loss_values.append(loss_function(x_update, y_update))
    
    x_initial, y_initial = x_update, y_update

# Graficar la función de pérdida en 2D
x_range = np.linspace(-2, 2, 400)
y_range = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x_range, y_range)
Z = loss_function(X, Y)
plt.figure(figsize=(10, 6))
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='Pérdida')
plt.scatter(x_values, y_values, color='red', s=20, label='Descenso de Gradiente')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Descenso de Gradiente para la Función de Pérdida\nx = ' + str(x_update) + '\ny = ' + str(y_update))
plt.legend()
plt.grid(True)
plt.show()