# Optimización y Análisis Estadístico de Modelos LSSVM

Este repositorio contiene scripts desarrollados en Python para la optimización, análisis de consistencia estadística y detección de valores atípicos en modelos **Least Squares Support Vector Machines (LSSVM)**, aplicados a problemas de predicción y regresión multivariable.

## Estructura del Proyecto

### 📁 Archivos Principales

- **`LSSVM_Optimizacion.py`**  
  Contiene el procedimiento completo para el entrenamiento, validación y optimización de un modelo LSSVM utilizando búsqueda por cuadrícula (grid search), normalización de variables y validación cruzada.

- **`outliers_LSSVM.py`**  
  Implementa pruebas estadísticas y gráficas para la detección de valores atípicos en los residuales del modelo, incluyendo pruebas como Grubbs, Shapiro-Wilk y visualizaciones tipo boxplot.

- **`Stadistic_Consostency.py`**  
  Evalúa la consistencia estadística del modelo LSSVM a través de métricas como el R², MSE, y pruebas de normalidad de residuos. También incluye análisis gráfico de dispersión y residuales.

## Requisitos

- Python ≥ 3.8  
- Librerías necesarias:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `scipy`
  - `statsmodels`

Puedes instalarlas con:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels
