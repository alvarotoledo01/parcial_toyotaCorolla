import os
import pickle
from dagster import AssetExecutionContext, asset
import mlflow
import pandas as pd
from sklearn.model_selection import KFold
import statsmodels.api as sm
import numpy as np


@asset(deps=["feature_engineering", "setup_mlflow"])
def train_model(
    context: AssetExecutionContext, feature_engineering: pd.DataFrame, setup_mlflow: str
):
    df = feature_engineering.copy()
    run_id = setup_mlflow

    y = df["Price"]
    X = df.drop(columns=["Price"])

    model_dir = os.path.join("mlartifacts", "model")
    os.makedirs(model_dir, exist_ok=True)

    with mlflow.start_run(run_id=run_id):

        k = 5
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        metrics = {
            "rmse": [],
            "mae": [],
            "mse_train": [],
            "mse_test": [],
            "r2_train": [],
            "r2_test": [],
            "adj_r2_train": [],
            "adj_r2_test": [],
            "aic": [],
            "bic": [],
        }

        fold = 0

        for train_index, test_index in kf.split(df):
            fold += 1
            context.log.info(f"Fold {fold}/{k}")

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            X_train = sm.add_constant(X_train)
            X_test = sm.add_constant(X_test)

            model = sm.OLS(y_train, X_train).fit()
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calcular métricas
            metrics["rmse"].append(np.sqrt(np.mean((y_test - y_test_pred) ** 2)))
            metrics["mae"].append(np.mean(np.abs(y_test - y_test_pred)))
            metrics["mse_train"].append(np.mean((y_train - y_train_pred) ** 2))
            metrics["mse_test"].append(np.mean((y_test - y_test_pred) ** 2))
            metrics["r2_train"].append(model.rsquared)

            # R2 test calculado explícitamente:
            ss_res = np.sum((y_test - y_test_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2_test = 1 - ss_res / ss_tot
            metrics["r2_test"].append(r2_test)

            metrics["adj_r2_train"].append(model.rsquared_adj)

            # Adj R2 test calculado igual que R2, con ajuste
            n_test = len(y_test)
            p = X_test.shape[1] - 1  # número de variables sin constante
            adj_r2_test = 1 - (1 - r2_test) * (n_test - 1) / (n_test - p - 1)
            metrics["adj_r2_test"].append(adj_r2_test)

            metrics["aic"].append(model.aic)
            metrics["bic"].append(model.bic)

        # Loguear métricas promedio y std
        for key in metrics:
            mlflow.log_metric(f"{key}_mean", np.mean(metrics[key]))
            mlflow.log_metric(f"{key}_std", np.std(metrics[key]))

        # Entrenar modelo final con todo el dataset
        X_full = sm.add_constant(X)
        final_model = sm.OLS(y, X_full).fit()

        # Guardar summary texto
        final_summary_path = os.path.join(model_dir, "final_model_summary.txt")
        with open(final_summary_path, "w") as f:
            f.write(str(final_model.summary()))
        mlflow.log_artifact(final_summary_path, "model")

        # Guardar modelo serializado con pickle
        final_model_path = os.path.join(model_dir, "final_model.pkl")
        with open(final_model_path, "wb") as f:
            pickle.dump(final_model, f)
        mlflow.log_artifact(final_model_path, "model")

        # Loguear parámetros clave
        mlflow.log_param("k_folds", k)

        context.log.info("Model training completed successfully.")

        return final_model
