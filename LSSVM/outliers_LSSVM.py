import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.covariance import MinCovDet

# --- 1. Carga y preparación de datos ---
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

# Separar características (X) y variable objetivo (y)
X_orig = df[["T", "Co", "Cte"]]
y = df["Qte"]

# Dividir datos en entrenamiento y prueba (guardando las características originales de prueba)
X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_orig, y, test_size=0.2, random_state=42)

# --- 2. Escalado de Datos ---
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_orig)
X_test = scaler.transform(X_test_orig)

# --- 3. Entrenamiento del modelo SVR ---
svm = SVR(kernel='rbf', C=100, gamma=1, epsilon=0.1)
svm.fit(X_train, y_train)

# --- 4. Predicción y Evaluación ---
y_pred = svm.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'RMSE del modelo: {rmse:.4f}\n')

# --- 5. Detección de Outliers en el conjunto de prueba ---
# Calcular residuos
residuals = y_test - y_pred

# Construir la matriz con los residuos y las variables independientes ORIGINALES del conjunto de prueba
hw_matrix = np.column_stack((residuals, X_test_orig.values))

# Aplicar el modelo de MCD para identificar los valores atípicos
mcd = MinCovDet().fit(hw_matrix)

# Calcular la distancia de Mahalanobis para cada punto del conjunto de prueba
dist = mcd.mahalanobis(hw_matrix)

# Identificar outliers como el 5% de los puntos con mayor distancia
# (puedes ajustar el percentil 95 según tus necesidades)
percentil = 95
outliers_indices_in_test = np.where(dist > np.percentile(dist, percentil))[0]

print('--- Detección de Outliers ---')
print(f'Forma de la matriz de entrada (Residuals + Features): {hw_matrix.shape}')
print(f'Se encontraron {len(outliers_indices_in_test)} outliers en el conjunto de prueba (usando el percentil {percentil}).')
print('Índices de los outliers (dentro del conjunto de prueba):', outliers_indices_in_test)
print('\nValores de los outliers ([Residuo, T, Co, Cte]):')
print(hw_matrix[outliers_indices_in_test])

# --- 6. Visualización de Resultados ---
# (Nota: El gráfico solo muestra una dimensión de X, 'Cte', para la visualización)
plt.figure(figsize=(10, 6))
# Graficar todos los puntos de prueba
plt.scatter(X_test[:, 2], y_test, color='black', label='Datos Reales (Prueba)')
# Graficar la línea de predicción
# Para una mejor visualización, ordenamos los puntos antes de trazar la línea
sort_idx = np.argsort(X_test[:, 2])
plt.plot(X_test[sort_idx, 2], y_pred[sort_idx], color='red', linewidth=3, label='Predicción SVR')
# Resaltar los outliers
plt.scatter(X_test[outliers_indices_in_test, 2], y_test.iloc[outliers_indices_in_test],
            color='lime', s=200, edgecolor='black', label='Outliers Detectados', zorder=5)
plt.xlabel('Cte (escalado)')
plt.ylabel('Qte')
plt.title('Predicción SVR vs Datos Reales (con Outliers)')
plt.legend()
plt.grid(True)
plt.show()

