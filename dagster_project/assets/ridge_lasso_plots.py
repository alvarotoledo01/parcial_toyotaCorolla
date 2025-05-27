# assets/ridge_lasso_plots.py
import os, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score
from dagster import AssetExecutionContext, asset
import mlflow

@asset(deps=["feature_engineering", "setup_mlflow"])
def ridge_lasso_plots(
    context: AssetExecutionContext,
    feature_engineering,
    setup_mlflow: str
) -> bool:
    df = feature_engineering.copy()
    X = df.drop(columns=["Price"]).values
    y = df["Price"].values

    alphas = np.logspace(3, -3, 100)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, Model in [("ridge", Ridge), ("lasso", Lasso)]:
        coefs, cv_mse = [], []
        for a in alphas:
            m = Model(alpha=a, max_iter=10000)
            m.fit(X, y)
            coefs.append(m.coef_)
            cv_mse.append(
                -cross_val_score(m, X, y, cv=kf, scoring="neg_mean_squared_error").mean()
            )

        out = os.path.join("mlartifacts", name)
        os.makedirs(out, exist_ok=True)

        # Trayectoria de coeficientes
        plt.figure(figsize=(8, 6))
        for i in range(coefs[0].shape[0]):
            plt.plot(np.log10(alphas), [c[i] for c in coefs])
        plt.title(f"{name.capitalize()} – Trayectoria de coeficientes")
        plt.xlabel("log10(alpha)")
        plt.ylabel("Coeficiente β")
        plt.grid(True)
        coef_path = os.path.join(out, f"{name}_coef_path.png")
        plt.savefig(coef_path)
        plt.close()

        # Curva CV-MSE
        plt.figure(figsize=(8, 6))
        plt.plot(np.log10(alphas), cv_mse, marker="o")
        plt.title(f"{name.capitalize()} – Curva CV MSE")
        plt.xlabel("log10(alpha)")
        plt.ylabel("MSE")
        plt.grid(True)
        cv_path = os.path.join(out, f"{name}_cv_error.png")
        plt.savefig(cv_path)
        plt.close()

        with mlflow.start_run(run_id=setup_mlflow):
            mlflow.log_artifact(coef_path, name)
            mlflow.log_artifact(cv_path, name)

    context.log.info("Gráficas completas de Ridge y Lasso generadas.")
    return True
