import pandas as pd

print("---------------------------------")
print("Carga de dataset:")
print("---------------------------------\n")


# Ruta del archivo local
ruta = "../datos/datos_ini.csv"

# Cargar dataset
df = pd.read_csv(ruta)

# Validar carga
print("Dimensiones:", df.shape)
print(df.head())

print("---------------------------------")
print("Análisis de estructura y tipos de datos:")
print("---------------------------------\n")

# Información general
df.info()

# Estadísticas descriptivas
print(df.describe(include="all").T)

# Tipos de datos
print(df.dtypes)

print("---------------------------------")
print("Identificación de problemas de calidad:")
print("---------------------------------\n")

# Valores nulos
print("Nulos por columna:")
print(df.isnull().sum())

# Duplicados
print("Registros duplicados:", df.duplicated().sum())

# Identificadores duplicados
print("IDs duplicados:", df["customerID"].duplicated().sum())

# Valores vacíos en TotalCharges
print(
    "TotalCharges vacíos:",
    df["TotalCharges"].astype(str).str.strip().eq("").sum()
)

# Distribución de la variable objetivo
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True) * 100)

print("---------------------------------")
print("Validación de reglas de negocio:")
print("---------------------------------\n")

# Conversión temporal para diagnóstico
total_charges = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("Valores no numéricos:", total_charges.isna().sum())

# Validaciones de negocio
print("Antigüedad negativa:", (df["tenure"] < 0).sum())
print("Cargos mensuales negativos:", (df["MonthlyCharges"] < 0).sum())
print("Cargos totales negativos:", (total_charges < 0).sum())

# Revisar registros con cargos faltantes
print(
    df.loc[
        total_charges.isna(),
        ["tenure", "TotalCharges", "Churn"]
    ]
)