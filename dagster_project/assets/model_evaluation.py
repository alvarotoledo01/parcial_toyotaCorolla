# assets/model_evaluation.py
from dagster import AssetExecutionContext, asset

@asset(deps=[
    "train_model",        # tu OLS original con k-folds
    "train_ridge_cv",
    "train_lasso_cv",
    "train_pcr_pls"
])
def model_evaluation(
    context: AssetExecutionContext,
    train_model,         # objeto statsmodels; metrics CV están en MLflow
    train_ridge_cv: dict,
    train_lasso_cv: dict,
    train_pcr_pls: dict
) -> dict:
    results = {}

    # OLS: recupera train vs test de MLflow manualmente o asume CV ~ test
    # Aquí simplificamos: si tu OLS CV rmse_mean (MLflow) es mucho menor que test_rmse_mean
    # pero como no retornas train/test en train_model, focalizamos en Ridge/Lasso.
    tr_r, te_r = train_ridge_cv["train_mse"], train_ridge_cv["test_mse"]
    ratio_r = te_r / tr_r
    results["Ridge"] = (
        "overfitting" if ratio_r > 1.2 else
        "underfitting" if ratio_r < 0.8 else
        "good_fit"
    )

    tr_l, te_l = train_lasso_cv["train_mse"], train_lasso_cv["test_mse"]
    ratio_l = te_l / tr_l
    results["Lasso"] = (
        "overfitting" if ratio_l > 1.2 else
        "underfitting" if ratio_l < 0.8 else
        "good_fit"
    )

    pcr_mse = train_pcr_pls["pcr_mse"]
    results["PCR"] = "underfitting" if pcr_mse > te_r else "good_fit"

    pls_mse = train_pcr_pls["pls_mse"]
    results["PLS"] = "underfitting" if pls_mse > te_r else "good_fit"

    for m, status in results.items():
        context.log.info(f"Evaluación {m}: {status}")

    return results
