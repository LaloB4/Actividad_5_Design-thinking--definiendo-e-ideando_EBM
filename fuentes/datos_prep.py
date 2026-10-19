
import pandas as pd
from pathlib import Path

# 1. Configuración de rutas
base = Path(__file__).resolve().parent
origen = base / "../datos/datos_ini.csv"
destino = base / "../datos/procesados/datos_limp.csv"

# 2. Cargar dataset
df = pd.read_csv(origen)
print("Registros originales:", len(df))

# 3. Eliminar duplicados
df = df.drop_duplicates(subset="customerID")

# 4. Convertir tipos y tratar valores nulos
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Imputar cero cuando tenure = 0
df.loc[
    (df["tenure"] == 0) & df["TotalCharges"].isna(),
    "TotalCharges"
] = 0

# 5. Validar reglas de negocio
servicios = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

inconsistencias = (
    ((df["InternetService"] == "No") &
     df[servicios].ne("No internet service").any(axis=1))
    |
    ((df["PhoneService"] == "No") &
     (df["MultipleLines"] != "No phone service"))
    |
    (df["tenure"] < 0)
    |
    (df["tenure"] > 72)
    |
    (df["MonthlyCharges"] < 0)
    |
    (df["TotalCharges"] < 0)
    |
    df.isna().any(axis=1)
    |
    ~df["Churn"].isin(["Yes", "No"])
)

# 6. Separar registros
df_invalidos = df[inconsistencias].copy()
df = df[~inconsistencias].copy()

# 7. Codificar variable objetivo
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 8. Guardar resultados
destino.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(destino, index=False)

if not df_invalidos.empty:
    df_invalidos.to_csv(
        destino.parent / "../datos_inconsistentes.csv",
        index=False
    )

# 9. Resumen
print("Registros preparados:", len(df))
print("Registros inconsistentes:", len(df_invalidos))
print("Valores nulos restantes:", df.isna().sum().sum())
print("Dataset guardado en:", destino)