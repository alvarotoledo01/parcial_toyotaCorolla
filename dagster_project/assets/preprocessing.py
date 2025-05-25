from dagster import asset
import mlflow
import pandas as pd


@asset(deps=["load_toyota_data", "setup_mlflow"])
def preprocess_data(
    context, load_toyota_data: pd.DataFrame, setup_mlflow: str
) -> pd.DataFrame:
    df = load_toyota_data.copy()
    original_shape = df.shape

    # Eliminar duplicados
    df = df.drop_duplicates()

    # Eliminar columnas innecesarias
    df = df.drop(columns=["Model"], axis=1)

    # One-hot encoding de variables categóricas
    categorial_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorial_cols, drop_first=True)

    # Logging en MLflow
    with mlflow.start_run(run_id=setup_mlflow):
        context.log.info(f"Forma original: {original_shape} → Nueva forma: {df.shape}")
        context.log.info(f"Número de features: {df.shape[1]}")

        mlflow.log_param("original_shape", str(original_shape))
        mlflow.log_param("preprocess_shape", str(df.shape))
        mlflow.log_param("n_features", df.shape[1])

    return df
