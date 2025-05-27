# assets/train_ridge_cv.py
import os, pickle
from dagster import AssetExecutionContext, asset
import mlflow
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import pandas as pd

@asset(deps=["feature_engineering", "setup_mlflow"])
def train_ridge_cv(
    context: AssetExecutionContext,
    feature_engineering: pd.DataFrame,
    setup_mlflow: str
) -> dict:
    df = feature_engineering.copy()
    run_id = setup_mlflow
    y = df["Price"].values
    X = df.drop(columns=["Price"]).values

    alphas = np.logspace(3, -3, 50)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge = RidgeCV(alphas=alphas, cv=kf).fit(X, y)

    # Métricas por fold
    train_mse, test_mse = [], []
    for train_i, test_i in kf.split(X):
        ridge.set_params(alpha=ridge.alpha_).fit(X[train_i], y[train_i])
        train_mse.append(((y[train_i] - ridge.predict(X[train_i]))**2).mean())
        test_mse.append(((y[test_i] - ridge.predict(X[test_i]))**2).mean())

    # Serializar modelo
    out = os.path.join("mlartifacts", "ridge_cv")
    os.makedirs(out, exist_ok=True)
    model_path = os.path.join(out, "ridge_cv_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(ridge, f)

    # Log en MLflow
    with mlflow.start_run(run_id=run_id, run_name="Ridge_CV"):
        mlflow.log_param("best_alpha", float(ridge.alpha_))
        mlflow.log_metric("ridge_train_mse_mean", float(np.mean(train_mse)))
        mlflow.log_metric("ridge_test_mse_mean", float(np.mean(test_mse)))
        mlflow.log_artifact(model_path, "model")

    context.log.info(
        f"RidgeCV α={ridge.alpha_} | train_mse={np.mean(train_mse):.3f} | test_mse={np.mean(test_mse):.3f}"
    )
    return {"train_mse": np.mean(train_mse), "test_mse": np.mean(test_mse), "alpha": ridge.alpha_}
