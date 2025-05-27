# assets/train_pcr_pls.py
import os, pickle
from dagster import AssetExecutionContext, asset
import mlflow
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_score


@asset(deps=["feature_engineering", "setup_mlflow"])
def train_pcr_pls(
    context: AssetExecutionContext, feature_engineering, setup_mlflow: str
) -> dict:
    df = feature_engineering.copy()
    y = df["Price"].values
    X = df.drop(columns=["Price"]).values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    run_id = setup_mlflow

    # PCR: elegir k óptimo
    best_mse, best_k = np.inf, None
    for k in range(1, X.shape[1] + 1):
        Xk = PCA(n_components=k).fit_transform(X)
        mse = -cross_val_score(
            LinearRegression(), Xk, y, cv=kf, scoring="neg_mean_squared_error"
        ).mean()
        if mse < best_mse:
            best_mse, best_k = mse, k
    pca = PCA(n_components=best_k).fit(X)
    reg = LinearRegression().fit(pca.transform(X), y)

    out1 = os.path.join("mlartifacts", "pcr")
    os.makedirs(out1, exist_ok=True)
    pickle.dump(pca, open(f"{out1}/pca.pkl", "wb"))
    pickle.dump(reg, open(f"{out1}/model.pkl", "wb"))

    # PLS: elegir n óptimo
    best2, best_n = np.inf, None
    for n in range(1, X.shape[1] + 1):
        mse2 = -cross_val_score(
            PLSRegression(n_components=n), X, y, cv=kf, scoring="neg_mean_squared_error"
        ).mean()
        if mse2 < best2:
            best2, best_n = mse2, n
    pls = PLSRegression(n_components=best_n).fit(X, y)
    out2 = os.path.join("mlartifacts", "pls")
    os.makedirs(out2, exist_ok=True)
    pickle.dump(pls, open(f"{out2}/model.pkl", "wb"))

    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("pcr_k", best_k)
        mlflow.log_metric("pcr_mse", float(best_mse))
        mlflow.log_param("pls_n", best_n)
        mlflow.log_metric("pls_mse", float(best2))
        mlflow.log_artifact(f"{out1}/pca.pkl", "pcr")
        mlflow.log_artifact(f"{out1}/model.pkl", "pcr")
        mlflow.log_artifact(f"{out2}/model.pkl", "pls")

    context.log.info(f"PCR k={best_k}, PLS n={best_n}")
    return {"pcr_mse": best_mse, "pls_mse": best2}
