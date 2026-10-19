# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)

# 1. Configuración de rutas
base = Path(__file__).resolve().parent
origen = base / "../datos/procesados/datos_limp.csv"
resultados = base / "resultados"

resultados.mkdir(parents=True, exist_ok=True)

# 2. Cargar dataset
df = pd.read_csv(origen)

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]

print("Registros:", len(df))
print("Variables predictoras:", X.shape[1])

# 3. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Preprocesamiento
numericas = ["tenure", "MonthlyCharges", "TotalCharges"]

categoricas = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

preprocesamiento = ColumnTransformer([
    ("num", StandardScaler(), numericas),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas)
], remainder="passthrough")

# 5. Definir modelos e hiperparámetros
modelos = {
    "Logistic_Regression": (
        LogisticRegression(max_iter=1000, random_state=42),
        {
            "modelo__C": [0.1, 1, 10]
        }
    ),
    "Random_Forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            "modelo__n_estimators": [100, 200],
            "modelo__max_depth": [5, 10, None],
            "modelo__min_samples_split": [2, 5]
        }
    )
}

# 6. Configurar MLflow
db_path = base / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment("Telco_Customer_Churn")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 7. Función para generar visualizaciones
def generar_visualizaciones(modelo, X_test, y_test, y_pred, y_prob, nombre):

    carpeta = resultados / nombre
    carpeta.mkdir(parents=True, exist_ok=True)

    # 7.1 Matriz de confusión
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["No Churn", "Churn"],
        cmap="Blues"
    )

    plt.title(f"Matriz de confusión - {nombre}")
    plt.savefig(carpeta / "matriz_confusion.png", bbox_inches="tight")
    plt.close()

    # 7.2 Curva ROC
    RocCurveDisplay.from_predictions(y_test, y_prob)

    plt.title(f"Curva ROC - {nombre}")
    plt.savefig(carpeta / "curva_roc.png", bbox_inches="tight")
    plt.close()

    # 7.3 Importancia de variables
    preprocesador = modelo.named_steps["preprocesamiento"]
    clasificador = modelo.named_steps["modelo"]

    variables = preprocesador.get_feature_names_out()

    if hasattr(clasificador, "feature_importances_"):
        importancia = clasificador.feature_importances_
    else:
        importancia = np.abs(clasificador.coef_[0])

    indices = np.argsort(importancia)[-15:]

    plt.figure(figsize=(10, 6))
    plt.barh(variables[indices], importancia[indices])
    plt.title(f"Variables más relevantes - {nombre}")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(carpeta / "importancia_variables.png", bbox_inches="tight")
    plt.close()

    # 7.4 Distribución de probabilidades
    y_real = np.asarray(y_test)

    plt.figure(figsize=(8, 5))

    plt.hist(
        y_prob[y_real == 0],
        bins=20,
        alpha=0.6,
        label="No Churn"
    )

    plt.hist(
        y_prob[y_real == 1],
        bins=20,
        alpha=0.6,
        label="Churn"
    )

    plt.axvline(
        0.5,
        color="red",
        linestyle="--",
        label="Umbral 0.5"
    )

    plt.xlabel("Probabilidad predicha de abandono")
    plt.ylabel("Número de clientes")
    plt.title(f"Distribución de probabilidades - {nombre}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        carpeta / "distribucion_probabilidades.png",
        bbox_inches="tight"
    )
    plt.close()

    # 7.5 Registrar visualizaciones en MLflow
    for archivo in carpeta.glob("*.png"):
        mlflow.log_artifact(
            str(archivo),
            artifact_path="visualizaciones"
        )

# 8. Entrenar y evaluar modelos
for nombre, (algoritmo, parametros) in modelos.items():

    print(f"\nEntrenando: {nombre}")

    pipeline = Pipeline([
        ("preprocesamiento", preprocesamiento),
        ("modelo", algoritmo)
    ])

    busqueda = GridSearchCV(
        pipeline,
        parametros,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    with mlflow.start_run(run_name=nombre):

        # Entrenamiento y optimización
        busqueda.fit(X_train, y_train)

        mejor_modelo = busqueda.best_estimator_

        # Predicciones
        y_pred = mejor_modelo.predict(X_test)
        y_prob = mejor_modelo.predict_proba(X_test)[:, 1]

        # Evaluación
        metricas = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "cv_f1": busqueda.best_score_
        }

        # Registrar parámetros y métricas
        mlflow.log_params(busqueda.best_params_)
        mlflow.log_metrics(metricas)

        # Generar y registrar gráficas
        generar_visualizaciones(
            mejor_modelo,
            X_test,
            y_test,
            y_pred,
            y_prob,
            nombre
        )

        # Registrar modelo
        mlflow.sklearn.log_model(
            sk_model=mejor_modelo,
            name="modelo",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        # Mostrar resultados
        print(f"\n=== {nombre} ===")
        print("Mejores parámetros:", busqueda.best_params_)

        for metrica, valor in metricas.items():
            print(f"{metrica}: {valor:.4f}")

print("\nEntrenamiento y registro en MLflow finalizados.")
