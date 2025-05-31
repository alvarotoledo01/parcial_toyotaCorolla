from dagster import Definitions, load_assets_from_modules
from dagstermill import ConfigurableLocalOutputNotebookIOManager

from dagster_project.assets import (
    feature_selection,
    lasso,
    ols,
    raw_dataset,
    clean_data,
    setup,
    residuals,
)

all_assets = load_assets_from_modules(
    [raw_dataset, clean_data, ols, feature_selection, lasso, setup, residuals]
)

defs = Definitions(
    assets=all_assets,
    resources={
        "output_notebook_io_manager": ConfigurableLocalOutputNotebookIOManager(
            base_dir="notebooks_outputs",
            asset_key_prefix=[],
        )
    },
)
