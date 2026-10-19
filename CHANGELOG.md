# Changelog

Registro de cambios y avances del proyecto **Telco Customer Churn**, orientado a la predicción del abandono de clientes mediante algoritmos de aprendizaje automático y herramientas de MLOps.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y utiliza versionado semántico.

## [1.0.0] - 2026-10-18

### Añadido
- Implementación del entrenamiento de dos algoritmos de clasificación: Logistic Regression y Random Forest.
- Script `fuentes/train.py` para entrenamiento, optimización y evaluación.
- Integración de MLflow para el seguimiento de experimentos.
- Configuración de SQLite (`mlflow.db`) como backend de MLflow.
- Optimización de hiperparámetros mediante `GridSearchCV`.
- Validación cruzada estratificada de cinco particiones.
- Registro de métricas: Accuracy, Precision, Recall, F1-score y ROC-AUC.
- Generación de visualizaciones:
  - Matriz de confusión.
  - Curva ROC.
  - Importancia de variables.
  - Distribución de probabilidades.
- Registro de modelos, métricas, parámetros y visualizaciones en MLflow.

### Resultados
- **Logistic Regression:** Accuracy de 80.48%, F1-score de 60.32% y ROC-AUC de 84.13%.
- **Random Forest:** Accuracy de 80.06%, F1-score de 58.00% y ROC-AUC de 84.04%.
- Selección preliminar de Logistic Regression como modelo recomendado por su mejor desempeño general.

### Corregido
- Sustitución del almacenamiento de seguimiento basado en archivos por SQLite para compatibilidad con MLflow.
- Configuración de tipos confiables en `skops` para permitir el registro del modelo Random Forest.

## [0.2.0] - Preparación de datos

### Añadido
- Script `fuentes/datos_prep.py` para limpieza y preparación del dataset.
- Conversión de `TotalCharges` a formato numérico.
- Tratamiento de valores nulos en clientes con antigüedad igual a cero.
- Validación de duplicados y reglas de negocio.
- Identificación de registros inconsistentes.
- Codificación de la variable objetivo `Churn` a valores binarios.
- Generación del archivo `datos/procesados/datos_limp.csv`.

### Resultados
- Registros originales: 7,043.
- Registros preparados: 7,043.
- Registros inconsistentes: 0.
- Valores nulos restantes: 0.

## [0.1.0] - Inicio del proyecto

### Añadido
- Definición del problema de predicción de abandono de clientes.
- Selección del dataset Telco Customer Churn.
- Configuración inicial del repositorio Git.
- Organización de directorios `datos/` y `fuentes/`.
- Script `fuentes/datos_info.py` para análisis exploratorio.
- Análisis de estructura, tipos de datos, valores nulos y distribución de clases.

### Resultados
- Identificación de 7,043 registros y 21 columnas.
- Distribución de la variable objetivo:
  - No Churn: 73.46%.
  - Churn: 26.54%.
- Identificación del desbalance de clases y selección de F1-score como métrica principal de optimización.

---

## Próximas mejoras

- Incorporar comparación estadística mediante la prueba de McNemar.
- Evaluar ajustes del umbral de clasificación.
- Analizar posibles mejoras en Recall para identificar más clientes con riesgo de abandono.
- Fortalecer el seguimiento y la reproducibilidad de experimentos.