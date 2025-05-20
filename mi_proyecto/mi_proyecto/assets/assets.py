from dagster import asset, AssetIn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import pandas as pd

@asset(
    group_name="clean_dataset",
    description="Carga el dataset de Toyota Corolla desde un archivo CSV"

)
def carga_dataset():
    url = "https://raw.githubusercontent.com/dodobeatle/dataeng-datos/refs/heads/main/ToyotaCorolla.csv"
    df_toyota = pd.read_csv(url)
    return df_toyota

@asset(
    group_name="clean_dataset",
    description="Limpieza de datos",
    ins={"df_toyota": AssetIn(key="carga_dataset")},

)
def limpieza(context, df_toyota):
    df_toyota_clean = df_toyota.drop(columns=["Id","Model","Fuel_Type"])
    return df_toyota_clean

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm

@asset(
    required_resource_keys={"mlflow"},
    group_name="clean_dataset",
    description="Transformación de datos y entrenamiento de regresión con métricas",
    ins={"df_toyota_clean": AssetIn(key="limpieza")},
)
def entrenar_modelo(context, df_toyota_clean):
    mlflow = context.resources.mlflow

    # 1. Separa X e y
    X = df_toyota_clean.drop(columns=["Price"])
    y = df_toyota_clean["Price"]

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Auto-logging en MLflow
    mlflow.autolog()

    # 4. Agrega el término constante (intercepto) a ambos sets
    X_train_const = sm.add_constant(X_train)
    X_test_const  = sm.add_constant(X_test)

    # 5. Ajusta el modelo
    model = sm.OLS(y_train, X_train_const).fit()
    context.log.info("Resumen del modelo:\n" + model.summary().as_text())

    # 6. Predice sobre el set de prueba
    y_pred = model.predict(X_test_const)

    # 7. Calcula métricas
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    # 8. Muestra en consola / logs
    context.log.info(f"MSE: {mse:.2f}")
    context.log.info(f"MAE: {mae:.2f}")
    context.log.info(f"R²:  {r2:.4f}")

    # 9. (Opcional) también las registra en MLflow
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2",  r2)

    return model


