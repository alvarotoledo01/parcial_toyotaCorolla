from dagster import asset


@asset(deps=["feature_engineering"])
def select_features(context, feature_engineering):
    df = feature_engineering.copy()

    # # Eliminar columnas innecesarias
    # df = df.drop(
    #     columns=[
    #         "Model",
    #         "Cylinders",
    #         "Id",
    #         "Fuel_Type_Diesel",
    #         "Fuel_Type_Petrol",
    #         "Mfg_Month",
    #         # "Mfg_Year",
    #         "BOVAG_Guarantee",
    #         "Mfr_Guarantee",
    #         "Guarantee_Period",
    #         "Gears",
    #         "Sport_Model",
    #         "Metallic_Rim",
    #         # "Met_Color",
    #         "Automatic",
    #         "CD_Player",
    #         "Radio",
    #         "Radio_cassette",
    #         "cc",
    #         "Doors",
    #         "Quarterly_Tax",
    #         # "Age_08_04",
    #         "Equipment_Count",
    #         "ABS",
    #         "Airbag_1",
    #         "Airbag_2",
    #         "Airco",
    #         "Automatic_airco",
    #         "Boardcomputer",
    #         "Central_Lock",
    #         "Powered_Windows",
    #         "Power_Steering",
    #         "Mistlamps",
    #         "Backseat_Divider",
    #         "Tow_Bar",
    #     ],
    #     axis=1,
    # )

    return df
