import os
from dagster import AssetExecutionContext, asset
from dotenv import load_dotenv
import mlflow


@asset
def setup_mlflow(context: AssetExecutionContext) -> str:
    load_dotenv()

    # Get MLflow configuration from environment
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    mlflow_run_name = os.getenv("MLFLOW_RUN_NAME", "ols_model_run")

    # Validate required configuration
    if not mlflow_tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is not set in the environment variables.")
    if not mlflow_experiment_name:
        raise ValueError(
            "MLFLOW_EXPERIMENT_NAME is not set in the environment variables."
        )

    # Configure MLflow
    context.log.info(
        f"Configuring MLflow: URI={mlflow_tracking_uri}, Experiment={mlflow_experiment_name}"
    )
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    # Start MLflow run
    context.log.info(f"Starting MLflow run: {mlflow_run_name}")
    with mlflow.start_run(run_name=mlflow_run_name) as run:
        run_id = run.info.run_id
        context.log.info(f"MLflow run started: {run_id}")
        return run_id
