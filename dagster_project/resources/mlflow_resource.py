from dagster import resource
import mlflow


@resource(config_schema={"tracking_uri": str, "experiment_name": str})
def mlflow_resource(init_context):
    tracking_uri = init_context.resource_config["tracking_uri"]
    experiment_name = init_context.resource_config["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    return mlflow
