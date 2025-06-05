import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, zscore
from scipy import stats
import math
import matplotlib.pyplot as plt
import seaborn as sns


def boxplot_feature(df, feature_name):
    """
    Boxplot para una feature numérica.
    """
    feature = df[feature_name]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(feature, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    ax.set_title(f"Boxplot de {feature_name}")
    ax.set_xlabel(feature_name)
    plt.tight_layout()
    plt.show()


def histogram_feature(df, feature_name, bins=30):
    """
    Histograma con KDE para una feature numérica.
    """
    feature = df[feature_name]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(feature, bins=bins, kde=True, ax=ax)
    ax.set_title(f"Distribución de {feature_name}")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()


def barplot_feature(df, feature_name):
    counts = df[feature_name].value_counts().sort_values(ascending=False)

    # Determine if we need to limit the number of categories shown
    max_categories = 20  # Maximum number of categories to display
    if len(counts) > max_categories:
        other_count = counts.iloc[max_categories:].sum()
        counts = counts.iloc[:max_categories]
        counts["Other"] = other_count

    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")

    # Add value labels on top of each bar
    for i, v in enumerate(counts):
        ax.text(i, v + 0.1, str(v), ha="center", fontweight="bold")

    ax.set_title(f"Distribution of {feature_name}")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def barplot_features_batch(
    df, feature_names, max_categories=20, batch_size=6, figsize=(15, 12)
):

    # Process features in batches
    for start_idx in range(0, len(feature_names), batch_size):
        end_idx = min(start_idx + batch_size, len(feature_names))
        batch_features = feature_names[start_idx:end_idx]

        # Calculate grid layout
        n_plots = len(batch_features)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

        # Create figure and axes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Create plots
        for i, feature_name in enumerate(batch_features):
            if feature_name not in df.columns:
                axes[i].text(
                    0.5,
                    0.5,
                    f"Column '{feature_name}' not found",
                    ha="center",
                    va="center",
                    color="red",
                )
                continue

            counts = df[feature_name].value_counts().sort_values(ascending=False)

            # Handle too many categories
            if len(counts) > max_categories:
                other_count = counts.iloc[max_categories:].sum()
                counts = counts.iloc[:max_categories].copy()
                counts["Other"] = other_count

            # Create bar plot
            counts.plot(kind="bar", ax=axes[i], color="skyblue", edgecolor="black")

            # Add value labels on top of each bar
            for j, v in enumerate(counts):
                axes[i].text(j, v + 0.1, str(v), ha="center", fontweight="bold")

            # Set titles and labels
            axes[i].set_title(f"Distribution of {feature_name}")
            axes[i].set_xlabel(feature_name)
            axes[i].set_ylabel("Count")
            axes[i].tick_params(axis="x", rotation=45, labelsize=8)

        # Remove unused subplots
        for j in range(n_plots, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


def violinplot_feature(df, feature_name, target_name="Price"):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Check if target is continuous, if so, discretize it
    if df[target_name].nunique() > 10:
        # Create categorical bins from continuous target
        df["target_bins"] = pd.qcut(
            df[target_name], 5, labels=[f"Q{i+1}" for i in range(5)]
        )
        sns.violinplot(
            x="target_bins", y=feature_name, data=df, ax=ax, inner="quartile"
        )
        ax.set_title(f"Distribution of {feature_name} by {target_name} (Quintiles)")
    else:
        # Use categorical target as is
        sns.violinplot(x=target_name, y=feature_name, data=df, ax=ax, inner="quartile")
        ax.set_title(f"Distribution of {feature_name} by {target_name}")

    ax.set_xlabel(target_name)
    ax.set_ylabel(feature_name)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def violinplot_features_batch(
    df, feature_names, target_name="Price", batch_size=4, figsize=(18, 12)
):

    # Check if target exists
    if target_name not in df.columns:
        raise ValueError(f"Target column '{target_name}' not found in DataFrame")

    # Create a temporary dataframe with binned target if needed
    temp_df = df.copy()
    target_is_continuous = temp_df[target_name].nunique() > 10

    if target_is_continuous:
        # Create categorical bins from continuous target
        temp_df["target_bins"] = pd.qcut(
            temp_df[target_name], 5, labels=[f"Q{i+1}" for i in range(5)]
        )
        target_col = "target_bins"
    else:
        target_col = target_name

    # Process features in batches
    for start_idx in range(0, len(feature_names), batch_size):
        end_idx = min(start_idx + batch_size, len(feature_names))
        batch_features = feature_names[start_idx:end_idx]

        # Calculate grid layout
        n_plots = len(batch_features)
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

        # Create figure and axes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Create plots
        for i, feature_name in enumerate(batch_features):
            if feature_name not in df.columns:
                axes[i].text(
                    0.5,
                    0.5,
                    f"Column '{feature_name}' not found",
                    ha="center",
                    va="center",
                    color="red",
                )
                continue

            try:
                # Create violin plot
                sns.violinplot(
                    x=target_col,
                    y=feature_name,
                    data=temp_df,
                    ax=axes[i],
                    inner="quartile",
                )

                # Set titles and labels
                if target_is_continuous:
                    axes[i].set_title(
                        f"Distribution of {feature_name} by {target_name} (Quintiles)"
                    )
                else:
                    axes[i].set_title(
                        f"Distribution of {feature_name} by {target_name}"
                    )

                axes[i].set_xlabel(target_name)
                axes[i].set_ylabel(feature_name)
                axes[i].tick_params(axis="x", rotation=45)

            except Exception as e:
                axes[i].text(
                    0.5, 0.5, f"Error: {str(e)}", ha="center", va="center", color="red"
                )
                axes[i].set_title(f"Failed: {feature_name}")

        # Remove unused subplots
        for j in range(n_plots, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

        # Clean up temporary column if created
        if target_is_continuous and "target_bins" in temp_df.columns:
            del temp_df["target_bins"]


def scatter_feature(df, feature_name, target_name="Price"):
    """
    Scatter plot de una feature contra una variable objetivo.
    """
    feature = df[feature_name]
    target = df[target_name]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(feature, target, alpha=0.6)
    ax.set_xlabel(feature_name)
    ax.set_ylabel(target_name)
    ax.set_title(f"{feature_name} vs {target_name}")
    plt.tight_layout()
    plt.show()


def plot_feature_analysis(df, feature_name, target_name="Price", bins=30):
    """
    Figura unificada con boxplot, histograma y scatter plot de una feature contra el target.
    """
    feature = df[feature_name]
    target = df[target_name]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Boxplot
    axs[0].boxplot(feature, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    axs[0].set_title(f"Boxplot de {feature_name}")
    axs[0].set_xlabel(feature_name)

    # Histograma con KDE
    sns.histplot(feature, bins=bins, kde=True, ax=axs[1])
    axs[1].set_title(f"Distribución de {feature_name}")
    axs[1].set_xlabel(feature_name)
    axs[1].set_ylabel("Frecuencia")

    # Scatter plot contra el target
    axs[2].scatter(feature, target, alpha=0.6)
    axs[2].set_xlabel(feature_name)
    axs[2].set_ylabel(target_name)
    axs[2].set_title(f"{feature_name} vs {target_name}")

    plt.tight_layout()
    plt.show()


def clean_outliers_zscore(feature: pd.Series, threshold: float = 2.0):
    """
    Remove outliers from a pandas Series using Z-score method.

    Parameters:
    - feature: pandas Series with data to clean
    - threshold: Z-score threshold (default=2.0)

    Returns:
    - cleaned_feature: Series with outliers removed
    - mask: Boolean mask identifying non-outlier values
    """
    feature_no_na = feature.dropna()
    z_scores = stats.zscore(feature_no_na)
    mask = abs(z_scores) < threshold

    # Create full-sized mask matching original feature (including NAs)
    full_mask = pd.Series(False, index=feature.index)
    full_mask.loc[feature_no_na.index] = mask

    cleaned_feature = feature[full_mask]

    return cleaned_feature, full_mask


def clean_outliers_iqr(feature: pd.Series):
    """
    Remove outliers from a pandas Series using IQR method.

    Parameters:
    - feature: pandas Series with data to clean

    Returns:
    - cleaned_feature: Series with outliers removed
    - mask: Boolean mask identifying non-outlier values
    """
    Q1 = feature.quantile(0.25)
    Q3 = feature.quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    mask = (feature >= lower_limit) & (feature <= upper_limit)
    cleaned_feature = feature[mask].copy()

    return cleaned_feature, mask


def resumen_outliers(df):

    # Filtrar solo columnas numéricas
    numericas = df.select_dtypes(include=[np.number])

    # Diccionarios para guardar los resultados
    outliers_iqr = {}
    outliers_zscore = {}

    for col in numericas.columns:
        # ----- Cálculo de IQR -----
        Q1 = numericas[col].quantile(0.25)
        Q3 = numericas[col].quantile(0.75)
        IQR = Q3 - Q1
        condicion_iqr = (numericas[col] < Q1 - 1.5 * IQR) | (
            numericas[col] > Q3 + 1.5 * IQR
        )
        outliers_iqr[col] = condicion_iqr.sum()

        # ----- Cálculo de Z-score (±2) -----
        col_z = numericas[col].dropna()
        zscores = zscore(col_z)
        condicion_z = (zscores < -2) | (zscores > 2)
        outliers_zscore[col] = condicion_z.sum()

    # Crear el DataFrame resumen
    df_resumen = pd.DataFrame(
        {
            "Outliers_IQR": outliers_iqr,
            "Outliers_Zscore": outliers_zscore,
        }
    )

    return df_resumen


def histogram_by_batch(df, batch_size=6):
    """
    Plot histograms for numeric columns in batches.

    Parameters:
    - df: pandas DataFrame
    - batch_size: number of histograms to display per figure
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    total_cols = len(numeric_cols)

    for start in range(0, total_cols, batch_size):
        end = min(start + batch_size, total_cols)
        subset = numeric_cols[start:end]
        n = len(subset)
        rows = math.ceil(n / 3)
        fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
        axes = axes.flatten()

        for i, col in enumerate(subset):
            data = df[col].dropna()

            try:
                if data.nunique() > 1:
                    axes[i].hist(
                        data,
                        bins=30,
                        alpha=0.7,
                        color="green",
                        edgecolor="black",
                        density=True,
                    )

                    # Add density curve
                    density = gaussian_kde(data)
                    x_vals = np.linspace(data.min(), data.max(), 1000)
                    axes[i].plot(x_vals, density(x_vals), color="red", linewidth=2)
                else:
                    axes[i].text(0.5, 0.5, "Constant value", ha="center", va="center")

                axes[i].set_title(col)
                axes[i].set_xlabel(col)

            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
                axes[i].set_title(col)

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


def boxplots_por_lotes(df, por_lote=6):
    columnas = df.select_dtypes(include=["number"]).columns
    print(columnas)
    total_columnas = len(columnas)

    for inicio in range(0, total_columnas, por_lote):
        fin = min(inicio + por_lote, total_columnas)
        subset = columnas[inicio:fin]
        n = len(subset)
        filas = math.ceil(n / 3)
        fig, axes = plt.subplots(filas, 3, figsize=(5 * 3, 4 * filas))
        axes = axes.flatten()

        for i, col in enumerate(subset):
            datos = df[col].dropna()

            try:
                datos = datos.astype(float)  # Forzamos conversión
                if datos.nunique() > 1:
                    axes[i].boxplot(
                        datos,
                        vert=False,
                        patch_artist=True,
                        boxprops=dict(facecolor="lightblue", color="blue"),
                        medianprops=dict(color="red"),
                    )
                else:
                    axes[i].text(0.5, 0.5, "Valor constante", ha="center", va="center")
                axes[i].set_title(col)
                axes[i].set_xlabel(col)

            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
                axes[i].set_title(col)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


import math
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def scatter_plots_by_batch(df, y_axis="Price", batch_size=6):
    """
    Create scatter plots between numeric columns and a target variable in batches.

    Parameters:
    - df: DataFrame containing the data
    - y_axis: Target variable to plot on y-axis (default: "price")
    - batch_size: Number of plots per figure
    """
    if y_axis not in df.columns:
        raise ValueError(f"The specified y_axis '{y_axis}' is not in the DataFrame.")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    columns_to_plot = [
        col for col in numeric_cols if col != y_axis and df[col].nunique() > 1
    ]

    for start in range(0, len(columns_to_plot), batch_size):
        end = min(start + batch_size, len(columns_to_plot))
        subset = columns_to_plot[start:end]

        cols = min(3, len(subset))
        rows = math.ceil(len(subset) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, col in enumerate(subset):
            try:
                x = df[col]
                y = df[y_axis]
                mask = x.notna() & y.notna()
                axes[i].scatter(
                    x[mask], y[mask], alpha=0.5, color="blue", edgecolors="w"
                )
                axes[i].set_xlabel(col)
                axes[i].set_ylabel(y_axis)
                axes[i].set_title(f"{col} vs {y_axis}")
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error: {str(e)}", ha="center", va="center")

        for j in range(len(subset), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


def bar_por_lotes(df, por_lote=6):
    columnas = df.select_dtypes(include=["number", "category", "object"]).columns

    total_columnas = len(columnas)

    for inicio in range(0, total_columnas, por_lote):
        fin = min(inicio + por_lote, total_columnas)
        subset = columnas[inicio:fin]
        n = len(subset)
        filas = math.ceil(n / 3)
        fig, axes = plt.subplots(filas, 3, figsize=(15, 4 * filas))
        axes = axes.flatten()

        for i, col in enumerate(subset):
            datos = df[col].dropna()

            try:
                # Si tiene muchas categorías únicas, se descarta
                if datos.nunique() > 50:
                    axes[i].text(
                        0.5, 0.5, "Demasiados valores únicos", ha="center", va="center"
                    )
                    axes[i].set_title(col)
                    continue

                # Conteo de frecuencias
                conteo = datos.value_counts().sort_index()

                axes[i].bar(conteo.index.astype(str), conteo.values, color="skyblue")
                axes[i].set_title(col)
                axes[i].set_xlabel("Valores")
                axes[i].set_ylabel("Frecuencia")
                axes[i].tick_params(axis="x", rotation=45)

            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
                axes[i].set_title(col)

        # Eliminar ejes vacíos
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def show_correlation_matrix(
    df,
    method: str = "pearson",
    show_plot: bool = True,
    annot: bool = True,
    figsize: tuple = (12, 10),
) -> pd.DataFrame:
    """
    Muestra y devuelve la matriz de correlación entre variables numéricas.

    Parámetros:
    - df: pandas.DataFrame con los datos.
    - method: str, tipo de correlación ('pearson', 'spearman', 'kendall').
    - show_plot: bool, si se muestra el gráfico de heatmap.
    - annot: bool, si se muestran los valores dentro del heatmap.
    - figsize: tuple, tamaño del gráfico.

    Devuelve:
    - DataFrame con la matriz de correlación.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        raise ValueError("El DataFrame no contiene columnas numéricas.")

    corr = numeric_df.corr(method=method)

    if show_plot:
        plt.figure(figsize=figsize)
        n_cols = len(corr.columns)
        font_scale = 0.9 if n_cols > 10 else 1.0
        sns.set(font_scale=font_scale)

        sns.heatmap(
            corr,
            annot=annot,
            fmt=".2f" if annot else None,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            annot_kws={"size": 7} if n_cols > 10 else {"size": 9},
        )
        plt.title(f"Matriz de correlación ({method.capitalize()})", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    return corr


def pares_correlacion_altas(corr_matrix, umbral=0.65):
    """
    Retorna un DataFrame con los pares de columnas con correlación absoluta >= umbral.

    Parámetros:
    - corr_matrix: Matriz de correlación (DataFrame).
    - umbral: Umbral mínimo absoluto de correlación (default=0.65).

    Retorna:
    - DataFrame con columnas: 'Variable_1', 'Variable_2', 'Correlación'
    """
    pares_altamente_correlacionados = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            correlacion = corr_matrix.iloc[i, j]
            if abs(correlacion) >= umbral:
                pares_altamente_correlacionados.append(
                    {"Variable_1": col1, "Variable_2": col2, "Correlación": correlacion}
                )

    df_resultado = pd.DataFrame(pares_altamente_correlacionados)
    df_resultado = df_resultado.sort_values(
        by="Correlación", key=lambda x: abs(x), ascending=False
    ).reset_index(drop=True)

    return df_resultado


def split(dataframe):
    x = dataframe.drop(columns=["price"])
    y = dataframe["price"]
    return x, y


def filtrar_columna_por_rango(serie: pd.Series, inferior: float, superior: float):
    """
    Filtra una Serie por valores dentro de un rango.

    Parámetros:
    - serie: Serie de pandas (columna del DataFrame)
    - inferior: límite inferior del rango
    - superior: límite superior del rango

    Retorna:
    - valores_filtrados: Serie con los valores dentro del rango
    - mascara: Serie booleana para aplicar al DataFrame original
    """
    mascara = (serie >= inferior) & (serie <= superior)
    valores_filtrados = serie[mascara]
    return valores_filtrados, mascara


def obtener_correlaciones_target(matriz_corr, variable_objetivo):
    """
    Devuelve un DataFrame con las correlaciones de 'variable_objetivo' respecto a todas las demás variables.

    Parámetros:
    - matriz_corr: pd.DataFrame, matriz de correlación de Pearson
    - variable_objetivo: str, nombre de la variable de interés

    Retorna:
    - DataFrame con dos columnas: ['variable', 'correlacion'], ordenado por correlación descendente
    """
    if variable_objetivo not in matriz_corr.columns:
        raise ValueError(
            f"La variable '{variable_objetivo}' no está en la matriz de correlación."
        )

    correlaciones = matriz_corr[variable_objetivo].drop(
        variable_objetivo
    )  # excluye correlación consigo misma

    return (
        correlaciones.to_frame(name="correlacion")
        .rename_axis("variable")
        .reset_index()
        .sort_values(by="correlacion", ascending=False)
    )


from scipy.stats import t, pearsonr


def correlacion_significativa(df, variable_objetivo, alpha=0.05):
    # Verificar que variable objetivo está en df
    if variable_objetivo not in df.columns:
        raise ValueError(
            f"La variable objetivo '{variable_objetivo}' no está en el DataFrame."
        )

    # Filtrar solo columnas numéricas (excluye la variable objetivo para evitar correlarse consigo misma)
    cols = df.select_dtypes(include=np.number).columns.drop(variable_objetivo)

    n = df.shape[0]
    df_gl = n - 2

    resultados = []

    for var in cols:
        # Calcular r y p usando pearsonr (para validar)
        r, p_pearson = pearsonr(df[variable_objetivo], df[var])

        # Calcular estadístico t según fórmula
        t_stat = r / math.sqrt((1 - r**2) / df_gl)

        # Valor crítico
        t_critico = t.ppf(1 - alpha / 2, df_gl)

        # Valor p desde t
        p_value = 2 * (1 - t.cdf(abs(t_stat), df_gl))

        # Decisión
        significativo = "Sí" if abs(t_stat) > t_critico else "No"

        resultados.append(
            {
                "variable": var,
                "r": r,
                "t_stat": t_stat,
                "p_value": p_value,
                "significativo": significativo,
            }
        )

        # Gráfico
        x = np.linspace(-4, 4, 500)
        y = t.pdf(x, df_gl)

        plt.figure(figsize=(8, 5))
        plt.plot(x, y, label=f"t-Student (df={df_gl})", color="black")
        plt.fill_between(
            x,
            y,
            where=(x <= -t_critico),
            color="red",
            alpha=0.3,
            label="Región de rechazo (α/2)",
        )
        plt.fill_between(x, y, where=(x >= t_critico), color="red", alpha=0.3)
        plt.axvline(
            t_stat, color="blue", linestyle="--", label=f"Estadístico t = {t_stat:.2f}"
        )
        plt.axvline(-t_stat, color="blue", linestyle="--")
        plt.axvline(
            t_critico,
            color="green",
            linestyle=":",
            label=f"T crítico = ±{t_critico:.2f}",
        )
        plt.axvline(-t_critico, color="green", linestyle=":")
        plt.title(f"Prueba t correlación: {variable_objetivo} vs {var}")
        plt.xlabel("t")
        plt.ylabel("Densidad de probabilidad")
        plt.legend()
        plt.grid(True)
        plt.show(block=False)

    return pd.DataFrame(resultados)


from scipy.stats import norm


def fisher_z_test(r, n, alpha=0.05):
    # Verifica que |r| < 1
    if abs(r) >= 1:
        raise ValueError("El coeficiente r debe estar estrictamente entre -1 y 1.")

    # Transformación z de Fisher
    z = 0.5 * np.log((1 + r) / (1 - r))

    # Error estándar
    se = 1 / np.sqrt(n - 3)

    # Estadístico z observado
    z_obs = z / se

    # Valor crítico para prueba bilateral
    z_crit = norm.ppf(1 - alpha / 2)

    # Valor p bilateral
    p_value = 2 * (1 - norm.cdf(abs(z_obs)))

    # Decisión
    decision = "significativa" if abs(z_obs) > z_crit else "NO significativa"

    # Resultados
    print(f"Transformación z de Fisher: {z:.4f}")
    print(f"Estadístico z observado: {z_obs:.4f}")
    print(f"Valor crítico z: ±{z_crit:.4f}")
    print(f"Valor p: {p_value:.4f}")
    print(f"➡ La correlación es {decision} al nivel α = {alpha}")

    return z_obs, p_value
