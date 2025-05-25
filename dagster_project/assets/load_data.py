from dagster import asset, AssetExecutionContext
import pandas as pd


@asset
def load_toyota_data(context: AssetExecutionContext) -> pd.DataFrame:
    context.log.info("Cargando datos de Toyota Corolla")

    # Cargar dataset
    df = pd.read_csv(
        "https://raw.githubusercontent.com/dodobeatle/dataeng-datos/refs/heads/main/ToyotaCorolla.csv",
        encoding="utf8",
        engine="python",
    )
    return df
