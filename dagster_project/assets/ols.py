import os
import pickle
from dagster import AssetExecutionContext, asset
import mlflow
import pandas as pd
from sklearn.model_selection import KFold
import statsmodels.api as sm
import numpy as np

from dagster_project.assets.config import MODEL_DIR


@asset(deps=["select_features"], required_resource_keys={"mlflow"})
def train_ols(context: AssetExecutionContext, select_features: pd.DataFrame):

    mlflow_resource: mlflow = context.resources.mlflow

    context.log.info("Entrenando modelo OLS")

    df = select_features.copy()

    context.log.info(df.dtypes)

    y = df["Price"]
    X = df.drop(columns=["Price"])

    with mlflow_resource.start_run(run_name="ols_model") as run:

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
            X_train = sm.add_constant(X_train)
            X_test = sm.add_constant(X_test)

            model = sm.OLS(y_train, X_train).fit()
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calcular métricas
            metrics["rmse_test"].append(np.sqrt(np.mean((y_test - y_test_pred) ** 2)))
            metrics["mae_test"].append(np.mean(np.abs(y_test - y_test_pred)))
            metrics["rmse_train"].append(
                np.sqrt(np.mean((y_train - y_train_pred) ** 2))
            )
            metrics["mae_train"].append(np.mean(np.abs(y_train - y_train_pred)))

            # R² test
            ss_res = np.sum((y_test - y_test_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2_test = 1 - ss_res / ss_tot

            # Adj R²
            metrics["r2_train"].append(model.rsquared_adj)
            n_test = len(y_test)
            p = X_test.shape[1] - 1  # sin constante
            adj_r2_test = 1 - (1 - r2_test) * (n_test - 1) / (n_test - p - 1)
            metrics["r2_test"].append(adj_r2_test)

        # Loguear métricas promedio
        for key in metrics:
            mlflow_resource.log_metric(f"{key}", round(float(np.mean(metrics[key])), 3))

        # Entrenar modelo final con todo el dataset
        X_full = sm.add_constant(X)
        final_model = sm.OLS(y, X_full).fit()

        # Loguear métricas del modelo final completo
        mlflow_resource.log_metric("r2", round(final_model.rsquared_adj, 4))

        # Log AIC and BIC metrics
        mlflow_resource.log_metric("aic", round(final_model.aic, 4))
        mlflow_resource.log_metric("bic", round(final_model.bic, 4))

        # Guardar resumen
        final_summary_path = os.path.join(MODEL_DIR, "ols_summary.txt")
        with open(final_summary_path, "w") as f:
            f.write(str(final_model.summary()))
        mlflow_resource.log_artifact(final_summary_path, "model")

        # Loguear parámetros
        mlflow_resource.log_param("ols_k_folds", k)
        mlflow_resource.log_param("ols_num_features", X.shape[1])

        context.log.info("OLS model training completed successfully.")

        return final_model, run.info.run_id
