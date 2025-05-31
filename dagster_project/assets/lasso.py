import os
from dagster import AssetExecutionContext, asset
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score


@asset(deps=["clean_data"], required_resource_keys={"mlflow"})
def train_lasso(context: AssetExecutionContext, clean_data):

    mlflow_resource: mlflow = context.resources.mlflow

    context.log.info("Entrenando modelo Lasso")

    df = pd.read_csv("data/clean_df.csv", encoding="utf8", engine="python")

    context.log.info(df.dtypes)

    X = df.drop(columns=["Price"])
    y = df["Price"]

    with mlflow_resource.start_run(run_name="lasso_model") as run:
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

            lasso_cv = LassoCV(cv=5, random_state=42)
            lasso_cv.fit(X_train, y_train)

            y_train_pred = lasso_cv.predict(X_train)
            y_test_pred = lasso_cv.predict(X_test)

            # Calcular métricas
            metrics["rmse_test"].append(
                np.sqrt(mean_squared_error(y_test, y_test_pred))
            )
            metrics["mae_test"].append(np.mean(np.abs(y_test - y_test_pred)))
            metrics["rmse_train"].append(
                np.sqrt(mean_squared_error(y_train, y_train_pred))
            )
            metrics["mae_train"].append(np.mean(np.abs(y_train - y_train_pred)))
            metrics["mse_train"].append(mean_squared_error(y_train, y_train_pred))
            metrics["mse_test"].append(mean_squared_error(y_test, y_test_pred))
            metrics["r2_train"].append(r2_score(y_train, y_train_pred))
            metrics["r2_test"].append(r2_score(y_test, y_test_pred))

        for metric, values in metrics.items():
            mean_value = np.mean(values)
            mlflow_resource.log_metric(metric, mean_value)
            context.log.info(f"{metric}: {mean_value}")

        final_model = LassoCV(cv=5, random_state=42)
        final_model.fit(X, y)
        mlflow_resource.sklearn.log_model(final_model, "lasso_model")

        mlflow_resource.log_param("alpha", final_model.alpha_)
        mlflow_resource.log_metric("r2", final_model.score(X, y))

        mlflow_resource.log_param("lasso_k_folds", k)
        mlflow_resource.log_param("lasso_coefs", final_model.coef_.tolist())

        return final_model, run.info.run_id
