from dagster import AssetExecutionContext, asset
import os

import mlflow

from dagster_project.assets.config import MODEL_DIR


@asset
def setup_mlflow(context: AssetExecutionContext):

    context.log.info("Setting up MLflow tracking")

    os.makedirs(MODEL_DIR, exist_ok=True)

    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    context.log.info(f"Using MLflow tracking URI: {mlflow_tracking_uri}")
    context.log.info(f"Using experiment name: {experiment_name}")
