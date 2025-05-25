from dagster import Definitions, load_assets_from_modules

from dagster_project.assets import (
    eda,
    feature_engineering,
    load_data,
    model_training,
    preprocessing,
    residuals,
    setup,
)  # noqa: TID252

all_assets = load_assets_from_modules(
    [
        load_data,
        setup,
        preprocessing,
        eda,
        feature_engineering,
        model_training,
        residuals,
    ]
)

defs = Definitions(
    assets=all_assets,
)
