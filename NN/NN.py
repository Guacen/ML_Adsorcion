import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Datos experimentales
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

# Escalado de los datos
scaler = MinMaxScaler()
X = scaler.fit_transform(df[["T", "Co", "Cte"]])
y = df["Qte"].values

# Crear modelo
def build_model(neurons=10, layers=1):
    model = Sequential()
    model.add(Dense(neurons, input_dim=3, activation='relu'))
    for _ in range(layers - 1):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Validación cruzada manual
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_score = float('inf')
best_config = None

for neurons in [18]:
    for layers in [4]:
        for epochs in [1000]:
            mse_scores = []
            for train_idx, val_idx in kf.split(X):
                model = build_model(neurons, layers)
                model.fit(X[train_idx], y[train_idx], epochs=epochs, batch_size=16, verbose=0)
                pred = model.predict(X[val_idx])
                mse_scores.append(mean_squared_error(y[val_idx], pred))
            avg_mse = np.mean(mse_scores)
            if avg_mse < best_score:
                best_score = avg_mse
                best_config = (neurons, layers, epochs)

# Entrenar modelo final
neurons, layers, epochs = best_config
final_model = build_model(neurons, layers)
final_model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
y_pred = final_model.predict(X).flatten()

# Métricas finales
mse_final = mean_squared_error(y, y_pred)
r2_final = r2_score(y, y_pred)


# Resultados
print("Configuración Óptima:")
print(f"  Neuronas: {neurons}")
print(f"  Capas Ocultas: {layers}")
print(f"  Épocas: {epochs}")
print(f"MSE Final: {mse_final}")
print(f"R² Final: {r2_final}")

residuals = y - y_pred
print(residuals)

# Crear un DataFrame
df_residuals = pd.DataFrame({'residuals': residuals})

# Guardar como CSV
df_residuals.to_csv('residuals.csv', index=False)

# Gráfico
plt.figure(figsize=(6,6))
plt.scatter(y, y_pred, alpha=0.7)
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
plt.xlabel("Qte Real")
plt.ylabel("Qte Predicho")
plt.title("Ajuste Lineal de la Red Neuronal")
plt.grid(True)
plt.tight_layout()
plt.show()

