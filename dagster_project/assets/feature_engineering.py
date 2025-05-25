from dagster import asset
import mlflow
import pandas as pd


@asset(deps=["preprocess_data", "setup_mlflow"])
def feature_engineering(
    context, preprocess_data: pd.DataFrame, setup_mlflow: str
) -> pd.DataFrame:
    df = preprocess_data.copy()

    # Equipment count
    equipment_cols = [
        "ABS",
        "Airbag_1",
        "Airbag_2",
        "Airco",
        "Automatic_airco",
        "Boardcomputer",
        "CD_Player",
        "Central_Lock",
        "Powered_Windows",
        "Power_Steering",
        "Radio",
        "Mistlamps",
        "Backseat_Divider",
        "Radio_cassette",
        "Tow_Bar",
    ]
    existing_equipment_cols = [col for col in equipment_cols if col in df.columns]
    df["Equipment_Count"] = df[existing_equipment_cols].sum(axis=1)
    context.log.info(
        f"Created equipment count feature from {len(existing_equipment_cols)} equipment columns"
    )

    # Eliminar columnas innecesarias
    df = df.drop(
        columns=[
            "Model",
            "Cylinders",
            "Id",
            "Fuel_Type_Diesel",
            "Fuel_Type_Petrol",
            "Mfg_Month",
        ]
        + equipment_cols,
        axis=1,
    )

    # Logear en mlflow
    with mlflow.start_run(run_id=setup_mlflow):
        mlflow.log_param("features", df.columns.tolist())

    return df
