from dagster import Definitions, load_assets_from_modules
from dagstermill import ConfigurableLocalOutputNotebookIOManager

from dagster_project.assets import raw_dataset, clean_data

all_assets = load_assets_from_modules([raw_dataset, clean_data])

defs = Definitions(
    assets=all_assets,
    resources={
        "output_notebook_io_manager": ConfigurableLocalOutputNotebookIOManager(
            base_dir="notebooks_outputs",
            asset_key_prefix=[],
        )
    },
)
