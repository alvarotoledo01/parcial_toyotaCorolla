from dagster import AssetExecutionContext, asset
import mlflow
import pandas as pd


@asset(deps=["load_toyota_data", "setup_mlflow"])
def preprocess_data(
    context: AssetExecutionContext, load_toyota_data: pd.DataFrame, setup_mlflow: str
) -> pd.DataFrame:
    df = load_toyota_data.copy()
    original_shape = df.shape

    # Eliminar duplicados
    df = df.drop_duplicates()

    # One-hot encoding de variables categóricas
    context.log.info("Realizando one-hot encoding de variables categóricas")
    columns_to_encode = ["Fuel_Type"]
    df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

    # Forzar columnas numérias
    context.log.info("Forzando columnas numéricas")
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    df = df.apply(pd.to_numeric, errors="coerce")
    context.log.info(df.dtypes)

    # Reordenar columnas
    cols = list(df.columns)
    cols.remove("Price")
    cols.append("Price")
    df = df[cols]

    # Logging en MLflow
    with mlflow.start_run(run_id=setup_mlflow):
        context.log.info(f"Forma original: {original_shape} → Nueva forma: {df.shape}")
        context.log.info(f"Número de features: {df.shape[1]}")

        mlflow.log_param("original_shape", str(original_shape))
        mlflow.log_param("preprocess_shape", str(df.shape))
        mlflow.log_param("n_features", df.shape[1])

    return df
