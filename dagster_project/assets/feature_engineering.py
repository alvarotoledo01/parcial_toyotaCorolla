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
        "Central_Lock",
        "Powered_Windows",
        "Power_Steering",
        "Mistlamps",
        "Backseat_Divider",
        "Tow_Bar",
    ]
    existing_equipment_cols = [col for col in equipment_cols if col in df.columns]
    df["Equipment_Count"] = df[existing_equipment_cols].sum(axis=1)
    context.log.info(
        f"Created equipment count feature from {len(existing_equipment_cols)} equipment columns"
    )

    # Equipment subcategories
    # Definir categorías de equipamiento
    safety_equipment = ["ABS", "Airbag_1", "Airbag_2"]
    comfort_equipment = ["Airco", "Automatic_airco", "Powered_Windows", "Central_Lock"]
    tech_equipment = ["Boardcomputer", "CD_Player", "Radio", "Radio_cassette"]

    # Crear subconteos por categoría
    safety_cols = [col for col in safety_equipment if col in df.columns]
    comfort_cols = [col for col in comfort_equipment if col in df.columns]
    tech_cols = [col for col in tech_equipment if col in df.columns]

    df["Safety_Score"] = df[safety_cols].sum(axis=1)
    df["Comfort_Score"] = df[comfort_cols].sum(axis=1)
    df["Tech_Score"] = df[tech_cols].sum(axis=1)

    context.log.info(
        f"Created categorical equipment scores for safety, comfort and tech"
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
            # "Mfg_Year",
            "BOVAG_Guarantee",
            "Mfr_Guarantee",
            "Guarantee_Period",
            "Gears",
            "Sport_Model",
            "Metallic_Rim",
            # "Met_Color",
            "Automatic",
            "CD_Player",
            "Radio",
            "Radio_cassette",
            "cc",
            "Doors",
            "Quarterly_Tax",
            # "Age_08_04",
            "Equipment_Count",
        ]
        + equipment_cols,
        axis=1,
    )

    # Logear en mlflow
    with mlflow.start_run(run_id=setup_mlflow):
        mlflow.log_param("features", df.columns.tolist())

    return df
