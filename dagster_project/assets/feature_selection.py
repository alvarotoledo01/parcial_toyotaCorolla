from dagster import AssetExecutionContext, asset
import pandas as pd


@asset(deps=["clean_data"], group_name="ols")
def select_features(context: AssetExecutionContext):
    df = pd.read_csv("data/clean_df.csv", encoding="utf8", engine="python")

    remove_features = [
        "Met_Color",
        "Airbag_1",
        "Airbag_2",
        "Doors",
        "Gears",
        "ABS",
        "Power_Steering",
        "Sport_Model",
        "Metallic_Rim",
        "Boardcomputer",
    ]

    df = df.drop(columns=remove_features, errors="ignore")

    return df
