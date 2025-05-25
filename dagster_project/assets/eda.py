from dagster import asset
import mlflow
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


@asset(deps=["preprocess_data", "setup_mlflow"])
def eda_report(context, preprocess_data: pd.DataFrame, setup_mlflow: str):
    df = preprocess_data.copy()
    eda_folder = os.path.join("mlartifacts", "eda")
    os.makedirs(eda_folder, exist_ok=True)

    # df describe
    context.log.info("Generating df describe")
    describe_path = os.path.join(eda_folder, "df_describe.txt")
    with open(describe_path, "w") as f:
        f.write(df.describe().round(2).to_string())

    # corelation matrix
    context.log.info("Generating correlation matrix")
    correlation_path = os.path.join(eda_folder, "correlation_matrix.png")
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"shrink": 0.75},
        linewidths=0.5,
        annot_kws={"size": 8},
    )
    plt.title("Correlation Matrix")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(correlation_path)
    plt.close()

    # pairplot
    context.log.info("Generating pairplot")
    pairplot_path = os.path.join(eda_folder, "pairplot.png")
    sns.pairplot(df, diag_kind="kde").savefig(pairplot_path)
    plt.close()

    # feature plots
    context.log.info("Generating feature plots")
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns
    features_folder = os.path.join(eda_folder, "features")
    os.makedirs(features_folder, exist_ok=True)

    feature_plots = []
    for feature in numerical_features:
        plt.figure(figsize=(14, 10))

        plt.subplot(2, 2, 1)
        sns.boxplot(x=df[feature])
        plt.title(f"Box Plot of {feature}")

        plt.subplot(2, 2, 2)
        sns.histplot(df[feature], kde=True)
        plt.title(f"Histogram of {feature}")

        plt.subplot(2, 2, 3)
        sns.violinplot(y=df[feature])
        plt.title(f"Violin Plot of {feature}")

        plt.subplot(2, 2, 4)
        sns.kdeplot(df[feature], fill=True)
        plt.title(f"Density Plot of {feature}")

        plt.tight_layout()

        feature_plot_path = os.path.join(features_folder, f"{feature}_plots.png")
        plt.savefig(feature_plot_path)
        plt.close()

        feature_plots.append(feature_plot_path)

    with mlflow.start_run(run_id=setup_mlflow):
        context.log.info(f"Logging EDA artifacts")
        mlflow.log_artifact(describe_path, "eda")
        mlflow.log_artifact(correlation_path, "eda")
        mlflow.log_artifact(pairplot_path, "eda")
        for feature_plot in feature_plots:
            mlflow.log_artifact(feature_plot, "eda/features")
        context.log.info("All artifacts logged successfully.")
