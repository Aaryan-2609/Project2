"""
Exploratory Data Analysis. Every function saves a figure to
reports/figures/ and returns the path, so run_pipeline.py can log
what was produced without holding figures in memory.
"""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


def _save(fig, name: str) -> str:
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return str(path)


def plot_severity_distributions(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, scale in zip(axes, config.TARGETS):
        order = config.SEVERITY_ORDER
        sns.countplot(x=f"{scale}_Severity", data=df, order=order, ax=ax, palette="viridis")
        ax.set_title(f"{scale} severity distribution")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _save(fig, "01_severity_distributions.png")


def plot_score_distributions(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, scale in zip(axes, config.TARGETS):
        sns.histplot(df[f"{scale}_Score"], bins=30, kde=True, ax=ax, color="steelblue")
        ax.set_title(f"{scale} raw score distribution")
    fig.tight_layout()
    return _save(fig, "02_score_distributions.png")


def plot_subscale_correlation(df: pd.DataFrame) -> str:
    score_cols = [f"{s}_Score" for s in config.TARGETS]
    corr = df[score_cols].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0, vmax=1, ax=ax)
    ax.set_title("Correlation between DASS subscale scores")
    return _save(fig, "03_subscale_correlation.png")


def plot_age_gender_vs_severity(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(x="Depression_Severity", y="age", data=df,
                order=config.SEVERITY_ORDER, ax=axes[0], palette="mako")
    axes[0].set_title("Age vs Depression severity")
    axes[0].tick_params(axis="x", rotation=30)

    gender_map = {1: "Male", 2: "Female", 3: "Other"}
    tmp = df.copy()
    tmp["gender_label"] = tmp["gender"].map(gender_map)
    sns.countplot(x="gender_label", hue="Depression_Severity", data=tmp,
                  hue_order=config.SEVERITY_ORDER, ax=axes[1], palette="viridis")
    axes[1].set_title("Gender vs Depression severity")
    fig.tight_layout()
    return _save(fig, "04_age_gender_vs_severity.png")


def plot_tipi_vs_severity(df: pd.DataFrame, tipi_features: pd.DataFrame) -> str:
    tmp = pd.concat([tipi_features, df["Depression_Severity"]], axis=1)
    melted = tmp.melt(id_vars="Depression_Severity", var_name="trait", value_name="score")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x="trait", y="score", hue="Depression_Severity", data=melted,
                hue_order=config.SEVERITY_ORDER, ax=ax, palette="viridis")
    ax.set_title("Big-Five personality traits vs Depression severity")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return _save(fig, "05_tipi_vs_depression.png")


def plot_missingness(df_raw: pd.DataFrame) -> str:
    frac_missing = df_raw.isna().mean().sort_values(ascending=False)
    frac_missing = frac_missing[frac_missing > 0].head(25)
    fig, ax = plt.subplots(figsize=(9, 6))
    frac_missing.plot(kind="barh", ax=ax, color="indianred")
    ax.set_title("Fraction missing per column (top 25)")
    ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig, "06_missingness.png")


def run_eda(df_raw: pd.DataFrame, df_labeled: pd.DataFrame, tipi_features: pd.DataFrame) -> list:
    figs = [
        plot_missingness(df_raw),
        plot_severity_distributions(df_labeled),
        plot_score_distributions(df_labeled),
        plot_subscale_correlation(df_labeled),
        plot_age_gender_vs_severity(df_labeled),
        plot_tipi_vs_severity(df_labeled, tipi_features),
    ]
    return figs
