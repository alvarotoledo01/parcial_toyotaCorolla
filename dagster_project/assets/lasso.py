import os
from dagster import AssetExecutionContext, asset
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


@asset(deps=["clean_data"], required_resource_keys={"mlflow"}, group_name="lasso")
def train_lasso(context: AssetExecutionContext, clean_data):
    mlflow_resource: mlflow = context.resources.mlflow
    context.log.info("Entrenando modelo Lasso")

    df = pd.read_csv("data/clean_df.csv", encoding="utf8", engine="python")
    context.log.info(df.dtypes)

    X = df.drop(columns=["Price"])
    y = df["Price"]

    # Dividir en train y test para evaluación
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow_resource.start_run(run_name="lasso_model") as run:
        # Entrenar LassoCV para encontrar el mejor alpha
        lasso_cv = LassoCV(cv=5, random_state=42, alphas=np.logspace(-6, 3, 100))
        lasso_cv.fit(X_train, y_train)
        best_alpha = lasso_cv.alpha_

        # Evaluar el modelo con el mejor alpha
        y_train_pred = lasso_cv.predict(X_train)
        y_test_pred = lasso_cv.predict(X_test)

        # Métricas
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Calcular métricas adicionales
        mae_train = np.mean(np.abs(y_train - y_train_pred))
        mae_test = np.mean(np.abs(y_test - y_test_pred))
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        # Loguear métricas
        mlflow_resource.log_metric("r2_train", r2_train)
        mlflow_resource.log_metric("r2_test", r2_test)
        mlflow_resource.log_metric("rmse_train", rmse_train)
        mlflow_resource.log_metric("rmse_test", rmse_test)
        mlflow_resource.log_metric("mae_train", mae_train)
        mlflow_resource.log_metric("mae_test", mae_test)
        mlflow_resource.log_metric("mse_train", mse_train)
        mlflow_resource.log_metric("mse_test", mse_test)
        mlflow_resource.log_param("best_alpha", best_alpha)

        # Generar gráfico de regularización
        alphas_to_try = np.logspace(-6, 3, 100)
        coefs = []

        for alpha in alphas_to_try:
            lasso = Lasso(alpha=alpha)
            lasso.fit(X_train, y_train)
            coefs.append(lasso.coef_.copy())

        coefs = np.array(coefs)

        # Crear gráfico
        plt.figure(figsize=(12, 8))
        for i, feature in enumerate(X.columns):
            plt.semilogx(alphas_to_try, coefs[:, i], label=feature)

        plt.axvline(
            x=best_alpha,
            color="k",
            linestyle="--",
            label=f"Mejor Alpha: {best_alpha:.6f}",
        )
        plt.xlabel("Alpha")
        plt.ylabel("Coeficientes")
        plt.title("Camino de regularización Lasso")
        plt.grid(True)

        # Si hay muchas características, mostrar leyenda fuera del gráfico
        if X.shape[1] > 10:
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            plt.legend(loc="best")

        plt.tight_layout()

        # Guardar y loguear el gráfico
        coef_path_file = "lasso_regularization_path.png"
        plt.savefig(coef_path_file)
        mlflow_resource.log_artifact(coef_path_file)
        plt.close()

        # Limpiar archivo temporal
        os.remove(coef_path_file)

        # Entrenar modelo final en todos los datos
        final_model = Lasso(alpha=best_alpha)
        final_model.fit(X, y)

        # Calcular y loguear R2 del modelo final
        final_predictions = final_model.predict(X)
        final_r2 = r2_score(y, final_predictions)
        mlflow_resource.log_metric("r2", final_r2)

        # Loguear coeficientes no nulos
        non_zero_coefs = {
            feature: coef
            for feature, coef in zip(X.columns, final_model.coef_)
            if coef != 0
        }
        context.log.info(f"Coeficientes no nulos: {non_zero_coefs}")
        mlflow_resource.log_param("non_zero_coefficients", non_zero_coefs)
        # Log non-zero coefficients as a text file artifact
        coefs_file = "non_zero_coefficients.txt"
        with open(coefs_file, "w") as f:
            f.write("Non-zero coefficients in Lasso model:\n\n")
            for feature, coef in sorted(
                non_zero_coefs.items(), key=lambda x: abs(x[1]), reverse=True
            ):
                f.write(f"{feature}: {coef:.6f}\n")
        mlflow_resource.log_artifact(coefs_file)
        os.remove(coefs_file)

        return final_model, run.info.run_id
