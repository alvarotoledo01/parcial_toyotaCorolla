from dagster import AssetExecutionContext, asset
import pandas as pd


@asset(deps=["clean_data"], group_name="ols")
def select_features(context: AssetExecutionContext):
    df = pd.read_csv("data/clean_df.csv", encoding="utf8", engine="python")

    keep_features = [
        "cc",
        "KM_Sqrt",
        "Age_08_04",
        "Boardcomputer",
        "Airco",
        "Powered_Windows",
        "HP",
        "High_Tax",
        "Weight",
    ]

    df = df[keep_features + ["Price"]]

    return df
