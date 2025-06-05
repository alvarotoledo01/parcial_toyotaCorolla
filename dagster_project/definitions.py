import os
from dagster import Definitions, load_assets_from_modules
from dagstermill import ConfigurableLocalOutputNotebookIOManager
from dagster_project.resources.mlflow_resource import mlflow_resource

from dagster_project.assets import (
    feature_selection,
    lasso,
    ols,
    raw_dataset,
    clean_data,
    residuals,
)

all_assets = load_assets_from_modules(
    [raw_dataset, clean_data, ols, feature_selection, lasso, residuals]
)

mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")

defs = Definitions(
    assets=all_assets,
    resources={
        "output_notebook_io_manager": ConfigurableLocalOutputNotebookIOManager(
            base_dir="notebooks_outputs",
            asset_key_prefix=[],
        ),
        "mlflow": mlflow_resource.configured(
            {
                "tracking_uri": mlflow_tracking_uri,
                "experiment_name": experiment_name,
            }
        ),
    },
)
