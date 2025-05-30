from dagster import AssetExecutionContext, asset
import mlflow
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold


@asset(deps=["clean_data"])
def train_pcr(context: AssetExecutionContext):

    context.log.info("Entrenando modelo PCR")

    df = pd.read_csv("data/clean_df.csv", encoding="utf8", engine="python")

    context.log.info(df.dtypes)

    X = df.drop(columns=["Price"])
    y = df["Price"]

    with mlflow.start_run(run_name="pcr_model"):
        k = 5
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        metrics = {
            "rmse_test": [],
            "mae_test": [],
            "rmse_train": [],
            "mae_train": [],
            "mse_train": [],
            "mse_test": [],
            "r2_train": [],
            "r2_test": [],
        }

        fold = 0

        for train_index, test_index in kf.split(df):
            fold += 1
            context.log.info(f"Fold {fold}/{k}")

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            pca = PCA(n_components=5)

            # Simulación de predicciones
            y_train_pred = X_train.mean(axis=1)
