# Predicción de abandono de clientes — Telco Customer Churn

Proyecto de clasificación supervisada para estimar el abandono de clientes (*churn*) de una empresa de telecomunicaciones. Se implementa un flujo reproducible de preparación de datos, entrenamiento y evaluación de dos algoritmos, búsqueda de hiperparámetros y seguimiento de experimentos con **MLflow**.

## Objetivo

Comparar **Regresión Logística** y **Random Forest** para predecir la variable `Churn` (1 = abandona, 0 = permanece), priorizando el equilibrio entre precisión y recuperación de clientes en riesgo mediante la métrica **F1-score**.

## Dataset

Se utiliza el dataset público **Telco Customer Churn**, con **7,043 registros** y **21 columnas** en su versión original.

Durante la preparación (`datos_prep.py`):

- Se eliminan duplicados por `customerID`, si existen.
- Se convierte `TotalCharges` a formato numérico y se asigna 0 a sus valores faltantes cuando `tenure = 0`.
- Se validan reglas de negocio para servicios contratados, antigüedad, cargos y variable objetivo.
- Se separan los registros inconsistentes para revisión, si los hubiera.
- Se convierte `Churn` de `Yes`/`No` a `1`/`0`.

**Resultado de la preparación:** 7,043 registros conservados, 0 inconsistencias y 0 valores nulos restantes.

> El archivo de entrada debe obtenerse de la fuente del dataset y colocarse en la ruta indicada a continuación. Verifica que el nombre y las columnas coincidan con los esperados por `datos_prep.py`.

## Estructura del proyecto

La organización real del repositorio es:

```text
Actividad_5_Design-thinking--definiendo-e-ideando_EBM/
├── datos/
│   ├── datos_ini.csv                 # Dataset original
│   └── procesados/
│       └── datos_limp.csv            # Dataset preparado
├── fuentes/
│   ├── datos_info.py                 # Inspección del dataset
│   ├── datos_prep.py                 # Limpieza y validaciones
│   ├── train.py                      # Entrenamiento y MLflow
│   ├── mlflow.db                     # Base de datos local de MLflow
│   ├── mlruns/                       # Archivos locales de MLflow
│   └── resultados/                   # Visualizaciones por modelo
├── README.md                         # Documentación del proyecto
└── CHANGELOG.md                      # Historial de cambios
```

La captura también muestra dos archivos con nombres `=4.0.0` y `=7.0.0`; no forman parte del flujo descrito y conviene revisar su origen antes de incluirlos en Git.

**Importante sobre las rutas:** `datos/` y `fuentes/` son carpetas hermanas. Por tanto, desde un script ubicado en `fuentes/`, la carpeta de datos se encuentra en `Path(__file__).resolve().parent.parent / "datos"`. Confirma que `datos_prep.py` y `train.py` utilizan esa ubicación; de otro modo, fallará la lectura de archivos.

## Requisitos

- **Python 3** (entorno utilizado: Python **3.13.3**).
- Git (para versionar el código).
- Librerías de Python:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `mlflow`
  - `skops` (serialización de modelos en la instalación de MLflow utilizada)

Instalación rápida:

```bash
python3 -m pip install pandas numpy scikit-learn matplotlib mlflow skops
```

Para registrar versiones exactas y facilitar la reproducción del entorno:

```bash
python3 -m pip freeze > requirements.txt
```

Si `requirements.txt` ya está incluido en el repositorio, instálalo en lugar de usar la instalación rápida:

```bash
python3 -m pip install -r requirements.txt
```

## Reproducción del experimento

### 1. Obtener el proyecto y preparar el entorno

```bash
git clone <URL_DEL_REPOSITORIO>
cd <CARPETA_DEL_REPOSITORIO>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

> Si todavía no existe `requirements.txt`, instala las librerías indicadas en **Requisitos**. En Windows, la activación del entorno virtual se realiza de forma distinta.

### 2. Colocar el dataset original

Guarda el archivo CSV original en la carpeta `datos/` de la raíz:

```text
datos/datos_ini.csv
```

### 3. Preparar los datos

```bash
python fuentes/datos_prep.py
```

La salida esperada de esta etapa es:

```text
Registros originales: 7043
Registros preparados: 7043
Registros inconsistentes: 0
Valores nulos restantes: 0
```

Se genera `datos/procesados/datos_limp.csv` en la raíz del repositorio.

### 4. Entrenar, evaluar y registrar los modelos

```bash
python fuentes/train.py
```

El script:

1. Carga el CSV procesado y excluye `customerID` de las variables predictoras.
2. Divide los datos en **80 % entrenamiento y 20 % prueba**, con `random_state=42` y estratificación por `Churn`.
3. Preprocesa las variables mediante `StandardScaler` y `OneHotEncoder`, dentro de un `Pipeline` para evitar fuga de información.
4. Entrena **Regresión Logística** y **Random Forest**.
5. Optimiza hiperparámetros con `GridSearchCV`, **StratifiedKFold de 5 particiones** y `scoring="f1"`.
6. Evalúa en el conjunto de prueba y registra métricas, parámetros, modelos y gráficas en **MLflow**.

La información de seguimiento se almacena en **SQLite** (`fuentes/mlflow.db`), configurada desde `train.py`.

### 5. Abrir MLflow

Desde la raíz del repositorio, ejecuta:

```bash
python -m mlflow ui --backend-store-uri sqlite:///fuentes/mlflow.db
```

Abre **http://127.0.0.1:5000** y selecciona el experimento **`Telco_Customer_Churn`**. Allí podrás comparar las ejecuciones `Logistic_Regression` y `Random_Forest`.

En cada ejecución, consulta **Artifacts → visualizaciones** para revisar las gráficas registradas.

## Configuración de modelos

| Modelo | Hiperparámetros explorados | Mejor configuración |
|---|---|---|
| Regresión Logística | `C`: 0.1, 1, 10 | `C=10` |
| Random Forest | `n_estimators`: 100, 200; `max_depth`: 5, 10, None; `min_samples_split`: 2, 5 | `n_estimators=200`, `max_depth=10`, `min_samples_split=2` |

La Regresión Logística utiliza `max_iter=1000`. Ambos algoritmos emplean `random_state=42`.

## Resultados

Resultados observados en el conjunto de prueba, excepto `cv_f1`, que corresponde al mejor promedio de F1 en validación cruzada:

| Métrica | Regresión Logística | Random Forest |
|---|---:|---:|
| Accuracy | **0.8048** | 0.8006 |
| Precision | 0.6552 | **0.6576** |
| Recall | **0.5588** | 0.5187 |
| F1-score | **0.6032** | 0.5800 |
| ROC-AUC | **0.8413** | 0.8404 |
| F1 validación cruzada (`cv_f1`) | **0.5972** | 0.5852 |

### Interpretación

**Regresión Logística** obtuvo el mayor **F1-score** (0.6032) y **Recall** (0.5588), por lo que es la opción recomendada entre las configuraciones evaluadas. **Random Forest** mostró una precisión ligeramente superior (0.6576 frente a 0.6552), pero identificó una menor proporción de clientes que abandonaron.

La diferencia observada es descriptiva; **no se realizó una prueba formal de significancia estadística**. La selección del modelo se basa en las métricas registradas, no en una afirmación de superioridad estadística concluyente.

## Visualizaciones y seguimiento en MLflow

Cada ejecución registra las siguientes gráficas en **Artifacts → visualizaciones**:

- `matriz_confusion.png`: aciertos y errores, incluidos falsos negativos.
- `curva_roc.png`: comportamiento del clasificador a distintos umbrales.
- `importancia_variables.png`: variables relevantes según coeficientes absolutos (Regresión Logística) o importancia por impureza (Random Forest).
- `distribucion_probabilidades.png`: distribución de probabilidades predichas para ambas clases.

Las importancias de ambos modelos se calculan de manera diferente y **no deben interpretarse como efectos causales ni compararse directamente en magnitud**.

## Conclusiones y trabajo futuro

- Se completó un flujo reproducible de preparación, entrenamiento, optimización, evaluación y registro de experimentos.
- La **Regresión Logística** fue el modelo con mejor equilibrio entre Precision y Recall según F1-score.
- El Recall de 0.5588 muestra margen de mejora en la detección de clientes que efectivamente abandonan.
- Como pasos futuros, se puede estudiar el ajuste del umbral de decisión, evaluar matrices de confusión y curvas Precision-Recall, y aplicar una comparación estadística formal de predicciones emparejadas (por ejemplo, McNemar para tasas de error).

## Notas de reproducibilidad

- Los comandos de esta guía se ejecutan desde la **raíz del repositorio**; los scripts están dentro de `fuentes/`.
- Utiliza el mismo CSV de origen, semillas aleatorias y versiones de librerías para facilitar la reproducción.
- Las ejecuciones adicionales generan nuevos *runs* en MLflow; no reemplazan los anteriores.
- Evita subir a Git archivos locales o generados como `.venv/`, `fuentes/mlflow.db`, `mlartifacts/`, `fuentes/mlruns/` y, si corresponde, `fuentes/resultados/`.
- Los resultados exactos pueden variar entre versiones de librerías y entornos de ejecución.
