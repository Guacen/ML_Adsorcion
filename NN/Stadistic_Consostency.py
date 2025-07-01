import numpy as np
from scipy import stats
import pandas as pd
import numpy as np
from math import comb
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kurtosis, norm, anderson

def swtest(x, alpha=0.05):
    x = np.asarray(x)
    x = x[~np.isnan(x)]  # Eliminar NaNs

    n = len(x)
    if n < 3:
        raise ValueError("El vector debe tener al menos 3 observaciones.")
    if n > 5000:
        print("Advertencia: El test de Shapiro-Wilk puede ser inexacto para n > 5000.")

    # Determinar si usar Shapiro-Wilk o Shapiro-Francia según la curtosis
    if kurtosis(x) > 3:
        # Shapiro-Francia: usar shapiro como aproximación
        stat, p = shapiro(x)
        W = stat
    else:
        # Shapiro-Wilk estándar
        stat, p = shapiro(x)
        W = stat

    H = int(p < alpha)
    return H, p, W

######################################################


def durbin_watson(residuales):
    diff_residuales = np.diff(residuales)
    numerador = np.sum(diff_residuales ** 2)
    denominador = np.sum(residuales ** 2)
    d = numerador / denominador
    return d


######################################################


# Función para calcular dL
def calculate_dL(residuos, k):
    n = len(residuos)
    alpha = 0.05
    critical_value = stats.f.ppf(1 - alpha/2, k, n - k - 1)
    return 2 - 2 * critical_value


######################################################


# Función para calcular dU
def calculate_dU(residuos, k):
    n = len(residuos)
    alpha = 0.05
    critical_value = stats.f.ppf(alpha/2, k, n - k - 1)
    return 2 - 2 * critical_value

######################################################

def calculate_randomness_probability(m,n, u_):
    C = comb(m+n,m) # CORREGIDO
    u_ += 1

    # Calcula la suma en la fórmula de la probabilidad de aleatoriedad
    sum_f = 0
    for u in range(2, u_+1):
        if u % 2 == 0:
            k = u // 2
            f_u = 2 * comb(m-1, k-1) * comb(n-1, k-1)
        else:
            k = (u+1) // 2
            f_u = comb(m-1, k-1) * comb(n-1, k-2) + comb(m-1, k-2) * comb(n-1, k-1)
        sum_f += f_u

    # Calcula la probabilidad de aleatoriedad
    p = 1/C * sum_f

    return p


#######################################################################

def adtest(residuals, alpha=0.05):
    residuals = np.asarray(residuals)
    residuals = residuals[~np.isnan(residuals)]  # Eliminar NaNs

    result = anderson(residuals, dist='norm')

    A2 = result.statistic
    critical_values = result.critical_values
    significance_levels = result.significance_level / 100  # convertir a proporciones

    # Buscar el valor crítico correspondiente al alpha solicitado
    if alpha in significance_levels:
        idx = np.where(significance_levels == alpha)[0][0]
        critical = critical_values[idx]
    else:
        # Si alpha no está en la lista exacta, interpolamos
        idx = np.searchsorted(significance_levels, alpha)
        if idx == 0:
            critical = critical_values[0]
        elif idx == len(critical_values):
            critical = critical_values[-1]
        else:
            # Interpolación lineal
            x0, x1 = significance_levels[idx-1], significance_levels[idx]
            y0, y1 = critical_values[idx-1], critical_values[idx]
            critical = y0 + (alpha - x0) * (y1 - y0) / (x1 - x0)

    # Decisión: si A² > valor crítico => rechazar H₀
    H = int(A2 > critical)

    return H, A2, critical

############################################

df = pd.read_csv('NN/residuals.csv')
Residuals = df['residuals'].values


d = durbin_watson(Residuals)

print(d)


#######################################################################


# Tu vector Qte
Qte = np.array([2.015, 3.971, 7.916, 9.360, 25.839, 36.820, 43.380, 48.360, 50.120, 50.232, 50.320, 51.134,
            1.941, 4.152, 6.184, 8.102, 10.165, 12.001, 14.388, 16.350, 19.366, 27.365, 39.293, 49.904, 54.653, 62.032, 65.164, 66.846, 66.234,
            1.687, 3.563, 8.162, 12.770, 26.794, 38.416, 47.758, 53.539, 58.700, 60.363, 61.979, 61.213,
            1.601, 3.804, 8.125, 16.840, 24.240, 40.278, 44.835, 51.844, 57.040, 58.769, 59.230, 59.386])

# Supón que ya tienes este vector de residuales (debes tener 56 elementos como en Qte)
# Residuals = np.array([...])

# Validación (opcional pero recomendable)
assert len(Qte) == len(Residuals), "Los vectores Qte y Residuals deben tener la misma longitud"

# Crear matriz combinada
combined = np.column_stack((Qte, Residuals))

# Ordenar por la primera columna (Qte)
combined_sorted = combined[np.argsort(combined[:, 0])]

# Extraer los residuales ordenados
Residuals_sorted = combined_sorted[:, 1]

# Análisis de signos
n = np.sum(Residuals_sorted > 0)
m = np.sum(Residuals_sorted <= 0)

# Cambios de signo
sec = 1
for i in range(1, len(Residuals_sorted)):
    if Residuals_sorted[i] * Residuals_sorted[i - 1] < 0:
        sec += 1

# Mostrar resultados
print("Positivos:", n)
print("Negativos o ceros:", m)
print("Cambios de signo (secciones):", sec)

P1 = calculate_randomness_probability(m, n, sec)

print("Probabilidad", 1-P1)


H, pValue, W = swtest(Residuals)

print("H =", H)
print("p-value =", pValue)
print("W =", W)

if H == 0:
    print("No se rechaza la hipótesis nula: la muestra parece provenir de una distribución normal.")
else:
    print("Se rechaza la hipótesis nula: la muestra no parece provenir de una distribución normal.")


H, A2, critical = adtest(Residuals)

print(f"Estadístico A²: {A2:.4f}")
print(f"Valor crítico (α=0.05): {critical:.4f}")
print(f"H = {H} → {'Se rechaza' if H else 'No se rechaza'} la hipótesis nula de normalidad.")


Qte_sorted = combined_sorted[:, 0]
Residuals_sorted = combined_sorted[:, 1]

plt.figure(figsize=(10, 5))
plt.plot(Qte_sorted, Residuals_sorted, marker='o', linestyle='-', color='steelblue')
plt.axhline(0, color='gray', linestyle='--')  # Línea horizontal en y=0 para ver cambios de signo
plt.title("Gráfico de Qte vs Residuales")
plt.xlabel("Qte")
plt.ylabel("Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()