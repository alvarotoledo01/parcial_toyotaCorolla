from dagster import AssetExecutionContext, asset
import pandas as pd


@asset(deps=["clean_data"], group_name="ols")
def select_features(context: AssetExecutionContext):
    df = pd.read_csv("data/clean_df.csv", encoding="utf8", engine="python")

    remove_features = [
        "cc",
        "Quarterly_Tax",
        "Gears",
        "Doors",
        "Met_Color",
        "Automatic",
        "BOVAG_Guarantee",
        "Guarantee_Period",
        "Sport_Model",
        "Metallic_Rim",
    ]

    df = df.drop(columns=remove_features)

    return df
