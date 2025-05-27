from dagster import Definitions, load_assets_from_modules


from dagster_project.assets import (
    eda,
    feature_engineering,
    load_data,
    model_evaluation,
    model_training,
    pca_plots,
    preprocessing,
    residuals,
    ridge_lasso_plots,
    setup,
    subset_stepwise_plots,
    train_lasso_cv,
    train_pcr_pls,
    train_ridge_cv,
)
from dagster_project.assets import feature_selection

all_assets = load_assets_from_modules(
    [
        eda,
        feature_engineering,
        feature_selection,
        load_data,
        model_evaluation,
        model_training,
        pca_plots,
        preprocessing,
        residuals,
        ridge_lasso_plots,
        setup,
        subset_stepwise_plots,
        train_lasso_cv,
        train_pcr_pls,
        train_ridge_cv,
    ]
)

defs = Definitions(
    assets=all_assets,
)
