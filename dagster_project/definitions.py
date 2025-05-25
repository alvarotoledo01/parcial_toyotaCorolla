from dagster import Definitions, load_assets_from_modules

from dagster_project.assets import load_data, preprocessing, setup  # noqa: TID252

all_assets = load_assets_from_modules([load_data, setup, preprocessing])

defs = Definitions(
    assets=all_assets,
)
