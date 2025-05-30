from dagstermill import define_dagstermill_asset
from dagster import AssetIn

clean_data = define_dagstermill_asset(
    name="clean_data",
    notebook_path="clean_data.ipynb",
    deps=["raw_dataset"],
)
