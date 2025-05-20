from dagster import Definitions, load_assets_from_modules
from dagster_mlflow import mlflow_tracking

from mi_proyecto.assets import assets  # noqa: TID252

all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=all_assets,
    resources={
        "mlflow": mlflow_tracking.configured({
            "mlflow_tracking_uri": "http://localhost:5000/",
            "experiment_name": "toyota_corolla_experiment"
        }),

    },
)
