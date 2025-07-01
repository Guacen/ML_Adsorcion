import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.covariance import MinCovDet

# Este código determina los outliers de la red neuronal. 

# Datos experimentales. 
data = {
    "T": [15]*13 + [25]*17 + [35]*13 + [45]*13,
    "Co": [5.218, 10.569, 19.965, 30.214, 48.076, 69.958, 97.345, 126.135, 149.765, 173.83, 198.5, 228.84, 261.38,
           5.197, 10.472, 15.485, 20.238, 25.358, 29.715, 35.689, 40.782, 48.48, 69.03, 100.225, 129.12, 147.685, 176.465, 206.09, 226.19, 247.87,
           5.329, 10.756, 20.943, 30.424, 49.365, 69.03, 100.225, 129.12, 143.875, 178.405, 186.08, 214.34, 240.33,
           5.377, 10.002, 20.986, 30.867, 48.887, 69.406, 107.695, 122.385, 149.345, 177.335, 199.1, 235.74, 251.24],
    "Cte": [0.704, 1.674, 2.233, 2.530, 4.718, 12.078, 19.353, 24.474, 35.764, 56.010, 85.980, 113.885, 146.840,
            0.850, 1.172, 1.633, 2.090, 2.589, 2.832, 3.460, 4.158, 5.100, 7.733, 12.208, 19.335, 26.263, 39.264, 62.362, 78.694, 103.985,
            1.15, 1.775, 2.660, 4.061, 7.088, 9.011, 16.174, 23.143, 31.187, 47.152, 59.346, 77.748, 107.692,
            1.291, 1.882, 2.787, 3.003, 5.692, 8.384, 18.473, 21.955, 31.734, 47.326, 69.698, 100.825, 118.215],
    "Qte": [2.015, 3.971, 7.916, 9.360, 14.360, 25.839, 36.820, 43.380, 48.360, 50.120, 50.232, 50.320, 51.134,
            1.941, 4.152, 6.184, 8.102, 10.165, 12.001, 14.388, 16.350, 19.366, 27.365, 39.293, 49.904, 54.653, 62.032, 65.164, 66.846, 66.234,
            1.687, 3.563, 8.162, 12.770, 21.870, 26.794, 38.416, 47.758, 53.539, 58.700, 60.363, 61.979, 61.213,
            1.601, 3.804, 8.125, 9.140, 16.840, 24.240, 40.278, 44.835, 51.844, 57.040, 58.769, 59.230, 59.386]
}

df = pd.DataFrame(data)
y = df["Qte"].values

df = pd.read_csv('Modelo Teorico/residuals.csv', sep=';')
residuals = df['residuals'].values

hw_matrix = np.column_stack((residuals, y))

# --- 3. Detección de Outliers (Tu código original sin cambios) ---
# El resto del código funciona igual, ya que opera sobre la matriz combinada.
# El algoritmo MCD calculará la distancia de Mahalanobis en este espacio de 4 dimensiones


# Aplicar el modelo de MCD para identificar los valores atípicos
mcd = MinCovDet().fit(hw_matrix)

# Calcular la distancia de Mahalanobis para cada punto
dist = mcd.mahalanobis(hw_matrix)

# Identificar outliers como el 5% de los puntos con mayor distancia
# (puedes ajustar el percentil 95 según tus necesidades)
outliers_indices = np.where(dist > np.percentile(dist, 95))[0]

print('--- Resultados ---')
print(f'Forma de la matriz de entrada (Residuals + Indep): {hw_matrix.shape}')
print(f'Se encontraron {len(outliers_indices)} outliers.')
print('Índices de los outliers:', outliers_indices)
print('\nValores de los outliers (Fila completa: [Residual, Var1, Var2, Var3]):')
print(hw_matrix[outliers_indices])