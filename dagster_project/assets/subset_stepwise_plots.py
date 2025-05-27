# assets/subset_stepwise_plots.py
import os, itertools, numpy as np, matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import KFold
from dagster import AssetExecutionContext, asset
import mlflow


@asset(deps=["feature_engineering", "setup_mlflow"])
def subset_stepwise_plots(
    context: AssetExecutionContext, feature_engineering, setup_mlflow: str
) -> bool:
    df = feature_engineering.copy()
    y = df["Price"].values
    Xdf = df.drop(columns=["Price"])
    cols = list(Xdf.columns)
    X = Xdf.values
    p = X.shape[1]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Best Subset CV-RMSE
    rmse_best = []
    for k in range(1, p + 1):
        errs = []
        for ti, vi in kf.split(X):
            best = max(
                itertools.combinations(range(p), k),
                key=lambda idxs: sm.OLS(y[ti], sm.add_constant(X[ti][:, idxs]))
                .fit()
                .rsquared_adj,
            )
            pred = (
                sm.OLS(y[ti], sm.add_constant(X[ti][:, best]))
                .fit()
                .predict(sm.add_constant(X[vi][:, best]))
            )
            errs.append((y[vi] - pred) ** 2)
        rmse_best.append(np.sqrt(np.mean(np.hstack(errs))))

    # Stepwise Forward CV-RMSE
    rmse_fw, included, rem = [], [], set(range(p))
    for _ in range(p):
        best_err, best_j = np.inf, None
        for j in rem:
            errs = []
            for ti, vi in kf.split(X):
                m = sm.OLS(y[ti], sm.add_constant(X[ti][:, [*included, j]])).fit()
                errs.append(
                    (y[vi] - m.predict(sm.add_constant(X[vi][:, [*included, j]]))) ** 2
                )
            err = np.mean(np.hstack(errs))
            if err < best_err:
                best_err, best_j = err, j
        included.append(best_j)
        rem.remove(best_j)
        rmse_fw.append(np.sqrt(best_err))

    out = os.path.join("mlartifacts", "subset_stepwise")
    os.makedirs(out, exist_ok=True)

    # Graficar Best Subset
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, p + 1), rmse_best, marker="o")
    plt.title("Best Subset CV-RMSE")
    plt.xlabel("Número de predictores (k)")
    plt.ylabel("RMSE")
    plt.grid(True)
    p1 = os.path.join(out, "best_subset_rmse.png")
    plt.savefig(p1)
    plt.close()

    # Graficar Stepwise Forward
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(rmse_fw) + 1), rmse_fw, marker="o", color="green")
    plt.title("Stepwise Forward CV-RMSE")
    plt.xlabel("Número de variables incluidas")
    plt.ylabel("RMSE")
    plt.grid(True)
    p2 = os.path.join(out, "stepwise_forward_rmse.png")
    plt.savefig(p2)
    plt.close()

    with mlflow.start_run(run_id=setup_mlflow):
        mlflow.log_artifact(p1, "subset_stepwise")
        mlflow.log_artifact(p2, "subset_stepwise")

    context.log.info("Gráficas de Best Subset y Stepwise generadas.")
    return True
