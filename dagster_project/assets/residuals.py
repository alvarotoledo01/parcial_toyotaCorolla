import os
from dagster import AssetExecutionContext, asset
import mlflow
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import ProbPlot
import numpy as np
from statsmodels.regression.linear_model import RegressionResultsWrapper


@asset(deps=["train_model", "feature_engineering", "setup_mlflow"])
def residual_analysis(
    context: AssetExecutionContext,
    train_model: RegressionResultsWrapper,
    feature_engineering: pd.DataFrame,
    setup_mlflow: str,
):
    df = feature_engineering.copy()
    model = train_model
    run_id = setup_mlflow

    # separar variables
    y = df["Price"]
    X = df.drop(columns=["Price"])
    X = sm.add_constant(X)

    # calcular valores ajustados y residuales
    y_pred = model.predict(X)
    residuals = y - y_pred
    standardized_residuals = model.get_influence().resid_studentized_internal
    context.log.info(f"Valores ajustados: {y_pred}")
    context.log.info(f"Residuales: {residuals}")

    # crear carpeta
    residuals_dir = os.path.join("mlartifacts", "residuals")
    os.makedirs(residuals_dir, exist_ok=True)

    # histograma de residuales
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    plt.axvline(x=0, color="r", linestyle="-")
    plt.xlabel("Residuales")
    plt.ylabel("Frecuencia")
    plt.title("Histograma de Residuales")
    plt.grid(True)
    hist_path = os.path.join(residuals_dir, "residuals_histogram.png")
    plt.savefig(hist_path)
    plt.close()

    # residuales vs valores ajustados
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color="r", linestyle="-")
    plt.xlabel("Valores Ajustados")
    plt.ylabel("Residuales")
    plt.title("Residuales vs Valores Ajustados")
    plt.grid(True)
    res_vs_fit_path = os.path.join(residuals_dir, "residuals_vs_fitted.png")
    plt.savefig(res_vs_fit_path)
    plt.close()

    # qq-plot
    plt.figure(figsize=(10, 6))
    QQ = ProbPlot(standardized_residuals)
    QQ.qqplot(line="45", alpha=0.5, color="#4C72B0", lw=1)
    plt.title("QQ-Plot de Residuales")
    plt.grid(True)
    qqplot_path = os.path.join(residuals_dir, "residuals_qqplot.png")
    plt.savefig(qqplot_path)
    plt.close()

    # scale location plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.6)
    plt.title("Scale-Location Plot")
    plt.xlabel("Valores Ajustados")
    plt.ylabel("√|Residuales Estandarizados|")
    plt.grid(True)
    scale_loc_path = os.path.join(residuals_dir, "scale_location_plot.png")
    plt.savefig(scale_loc_path)
    plt.close()

    # residuals vs leverage plot
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag

    plt.figure(figsize=(10, 6))
    plt.scatter(leverage, standardized_residuals, alpha=0.6)
    plt.axhline(y=0, color="r", linestyle="-")
    plt.xlabel("Leverage")
    plt.ylabel("Residuales Estandarizados")
    plt.title("Residuales vs Leverage")
    plt.grid(True)
    res_vs_lev_path = os.path.join(residuals_dir, "residuals_vs_leverage.png")
    plt.savefig(res_vs_lev_path)
    plt.close()

    with mlflow.start_run(run_id=run_id):
        # Logear los gráficos
        context.log.info("Logging residuals plots to MLflow")
        mlflow.log_artifact(hist_path, "residuals")
        mlflow.log_artifact(res_vs_fit_path, "residuals")
        mlflow.log_artifact(qqplot_path, "residuals")
        mlflow.log_artifact(scale_loc_path, "residuals")
        mlflow.log_artifact(res_vs_lev_path, "residuals")
        context.log.info("Residuals analysis completed and logged to MLflow")
