import os
from dagster import AssetExecutionContext, asset
import pandas as pd


@asset()
def raw_dataset(context: AssetExecutionContext) -> str:
    context.log.info("Cargando datos de Toyota Corolla")

    # Cargar dataset
    df = pd.read_csv(
        "https://raw.githubusercontent.com/dodobeatle/dataeng-datos/refs/heads/main/ToyotaCorolla.csv",
        encoding="utf8",
        engine="python",
    )

    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    df_path = os.path.join(data_dir, "raw_df.csv")
    df.to_csv(df_path, index=False)

    return df_path
