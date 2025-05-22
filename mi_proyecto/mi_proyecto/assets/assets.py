from dagster import asset, AssetIn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

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



@asset(
    required_resource_keys={"mlflow"},
    group_name="clean_dataset",
    description="Transformación de datos y entrenamiento de regresión con métricas",
    ins={"df_toyota_clean": AssetIn(key="limpieza")},
)
def entrenar_modelo(context, df_toyota_clean):
    mlflow = context.resources.mlflow

    # ——————————————— 1. Separa X e y ———————————————
    X = df_toyota_clean.drop(columns=["Price"])
    y = df_toyota_clean["Price"]

    # ——————————————— 2. Split ———————————————
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ——————————————— 3. Autolog MLflow ———————————————
    mlflow.autolog()

    # ——————————————— 4. Añade intercepto ———————————————
    X_train_const = sm.add_constant(X_train)
    X_test_const  = sm.add_constant(X_test)

    # ——————————————— 5. Entrena modelo OLS ———————————————
    model = sm.OLS(y_train, X_train_const).fit()
    context.log.info("Resumen del modelo:\n" + model.summary().as_text())

    # ——————————————— 6. Predicción y métricas ———————————————
    y_pred = model.predict(X_test_const)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    context.log.info(f"MSE: {mse:.2f}  MAE: {mae:.2f}  R²: {r2:.4f}")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2",  r2)

    # ——————————————— 7. Matriz de correlación anotada ———————————————
    corr = df_toyota_clean.corr()
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    cax = ax1.matshow(corr, vmin=-1, vmax=1)
    fig1.colorbar(cax, fraction=0.046, pad=0.04)
    cols = corr.columns
    ax1.set_xticks(range(len(cols)))
    ax1.set_yticks(range(len(cols)))
    ax1.set_xticklabels(cols, rotation=90, fontsize=8)
    ax1.set_yticklabels(cols,         fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iat[i, j]
            color = "white" if abs(val)>0.5 else "black"
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)
    plt.title("Matriz de correlación")
    plt.tight_layout()
    path_corr = "correlation_matrix.png"
    fig1.savefig(path_corr, dpi=150)
    plt.close(fig1)
    mlflow.log_artifact(path_corr)

    # ——————————————— 8. Residuales vs Valores Ajustados ———————————————
    residuales = y_train - model.predict(X_train_const)
    fitted     = model.predict(X_train_const)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(fitted, residuales, s=10, alpha=0.6)
    ax2.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax2.set_xlabel("Valores ajustados")
    ax2.set_ylabel("Residuales")
    ax2.set_title("Residuales vs Valores Ajustados")
    plt.tight_layout()
    path_resid = "residuals_vs_fitted.png"
    fig2.savefig(path_resid, dpi=150)
    plt.close(fig2)
    mlflow.log_artifact(path_resid)

        # ——————————————— 9. Scatter plot para detectar outliers en cada feature ———————————————
    for col in df_toyota_clean.columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.scatter(df_toyota_clean.index, df_toyota_clean[col], s=10, alpha=0.6)
        ax.set_xlabel("Índice")
        ax.set_ylabel(col)
        ax.set_title(f"Scatter de {col} vs Índice")
        plt.tight_layout()

        img_name = f"scatter_{col}.png"
        fig.savefig(img_name, dpi=100)
        plt.close(fig)

        # Sube cada gráfico como artifact
        mlflow.log_artifact(img_name)


    return model


