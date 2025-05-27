# assets/train_lasso_cv.py
import os, pickle
from dagster import AssetExecutionContext, asset
import mlflow
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
import pandas as pd

@asset(deps=["feature_engineering", "setup_mlflow"])
def train_lasso_cv(
    context: AssetExecutionContext,
    feature_engineering: pd.DataFrame,
    setup_mlflow: str
) -> dict:
    df = feature_engineering.copy()
    run_id = setup_mlflow
    y = df["Price"].values
    X = df.drop(columns=["Price"]).values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lasso = LassoCV(cv=kf, random_state=42, max_iter=10000).fit(X, y)

    train_mse, test_mse = [], []
    for train_i, test_i in kf.split(X):
        lasso.set_params(alpha=lasso.alpha_).fit(X[train_i], y[train_i])
        train_mse.append(((y[train_i] - lasso.predict(X[train_i]))**2).mean())
        test_mse.append(((y[test_i] - lasso.predict(X[test_i]))**2).mean())

    out = os.path.join("mlartifacts", "lasso_cv")
    os.makedirs(out, exist_ok=True)
    model_path = os.path.join(out, "lasso_cv_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(lasso, f)

    with mlflow.start_run(run_id=run_id, run_name="Lasso_CV"):
        mlflow.log_param("best_alpha", float(lasso.alpha_))
        mlflow.log_metric("lasso_train_mse_mean", float(np.mean(train_mse)))
        mlflow.log_metric("lasso_test_mse_mean", float(np.mean(test_mse)))
        mlflow.log_artifact(model_path, "model")

    context.log.info(
        f"LassoCV α={lasso.alpha_} | train_mse={np.mean(train_mse):.3f} | test_mse={np.mean(test_mse):.3f}"
    )
    return {"train_mse": np.mean(train_mse), "test_mse": np.mean(test_mse), "alpha": lasso.alpha_}
