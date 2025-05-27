# assets/pca_plots.py
import os, numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dagster import AssetExecutionContext, asset
import mlflow


@asset(deps=["feature_engineering", "setup_mlflow"])
def pca_plots(
    context: AssetExecutionContext, feature_engineering, setup_mlflow: str
) -> bool:
    df = feature_engineering.drop(columns=["Price"])
    X = df.values
    cols = df.columns.tolist()
    out = os.path.join("mlartifacts", "pca")
    os.makedirs(out, exist_ok=True)

    # Varianza acumulada
    pca_full = PCA().fit(X)
    cum_var = pca_full.explained_variance_ratio_.cumsum()
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o")
    plt.title("PCA – Varianza Acumulada")
    plt.xlabel("Número de componentes")
    plt.ylabel("Varianza Acumulada")
    plt.grid(True)
    p1 = os.path.join(out, "cum_variance.png")
    plt.savefig(p1)
    plt.close()

    # Biplot PC1 vs PC2
    pca2 = PCA(n_components=2).fit(X)
    Z = pca2.transform(X)
    L = pca2.components_.T
    plt.figure(figsize=(8, 8))
    plt.scatter(Z[:, 0], Z[:, 1], alpha=0.5)
    for i, var in enumerate(cols):
        plt.arrow(
            0,
            0,
            L[i, 0] * max(Z[:, 0]),
            L[i, 1] * max(Z[:, 1]),
            color="r",
            head_width=0.05,
        )
        plt.text(
            L[i, 0] * max(Z[:, 0]) * 1.1,
            L[i, 1] * max(Z[:, 1]) * 1.1,
            var,
            color="r",
            fontsize="small",
        )
    plt.title("PCA Biplot")
    plt.grid(True)
    p2 = os.path.join(out, "biplot.png")
    plt.savefig(p2)
    plt.close()

    # Loadings PC1 / PC2
    plt.figure(figsize=(10, 6))
    idx = np.arange(len(cols))
    w = 0.35
    plt.bar(idx, L[:, 0], w, label="PC1")
    plt.bar(idx + w, L[:, 1], w, label="PC2")
    plt.xticks(idx + w / 2, cols, rotation=90, fontsize="small")
    plt.title("Loadings PC1 y PC2")
    plt.legend()
    plt.tight_layout()
    p3 = os.path.join(out, "loadings.png")
    plt.savefig(p3)
    plt.close()

    with mlflow.start_run(run_id=setup_mlflow):
        mlflow.log_artifact(p1, "pca")
        mlflow.log_artifact(p2, "pca")
        mlflow.log_artifact(p3, "pca")

    context.log.info("Gráficas PCA generadas.")
    return True
