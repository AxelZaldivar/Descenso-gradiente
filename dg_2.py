import matplotlib.pyplot as plt
import numpy as np
import csv

# Inicializar listas, variables y parámetros
years_of_experience = []
salary = []
m = 0
b = 0
learning_rate = 0.01
iteraciones = 1000

# Leer el archivo CSV y extraer las columnas
with open('Salary_dataset.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        years_of_experience.append(float(row['YearsExperience']))
        salary.append(float(row['Salary']))
years_of_experience = np.array(years_of_experience)

# Función de costo
def mse_loss(y, y_1):
    return np.mean((y - y_1) ** 2)

# Descenso de gradiente
for iteration in range(iteraciones):
    y_1 = m * years_of_experience + b
    gradient_m = np.mean(years_of_experience * (salary - y_1)) * -2
    gradient_b = np.mean(salary - y_1) * -2
    m = m - learning_rate * gradient_m
    b = b - learning_rate * gradient_b
    mse_loss(salary, y_1)

# Crear la línea de regresión lineal
linea = m * years_of_experience + b

print(f"Pendiente (m): {m}")
print(f"Intersección (b): {b}")

# Grafica los datos y la línea de regresión
plt.scatter(years_of_experience, salary)
plt.plot(years_of_experience, linea, color='red')
plt.xlabel('Años de experiencia')
plt.ylabel('Salario')
plt.legend()
plt.title('Regresión lineal.\nValor de m: ' + str(m) + '\nValor de b: ' + str(b))
plt.show()