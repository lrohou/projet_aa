"""
Fonctions de visualisation pour ReviewScope.
==============================================
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def save_fig(fig, name, dpi=150):
    """Sauvegarde une figure dans le dossier reports/figures/."""
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Figure sauvegardée : {path}")
    plt.close(fig)


def plot_score_distribution(df, save=True):
    """Distribution des notes (1-5)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["Score"].value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, color=sns.color_palette("husl", 5))
    ax.set_xlabel("Note")
    ax.set_ylabel("Nombre d'avis")
    ax.set_title("Distribution des notes")
    ax.set_xticks([1, 2, 3, 4, 5])
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f"{count:,}", ha="center", va="bottom", fontsize=9)
    if save:
        save_fig(fig, "score_distribution")
    return fig


def plot_helpfulness_distribution(df, save=True):
    """Distribution du ratio d'utilité."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    has_votes = df[df["HasVotes"] == True]["HelpfulnessRatio"]
    axes[0].hist(has_votes, bins=50, edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Ratio d'utilité")
    axes[0].set_ylabel("Fréquence")
    axes[0].set_title("Distribution du ratio d'utilité\n(avis avec au moins 1 vote)")
    axes[0].axvline(x=config.HELPFULNESS_THRESHOLD, color="red", linestyle="--",
                    label=f"Seuil = {config.HELPFULNESS_THRESHOLD}")
    axes[0].legend()

    axes[1].hist(df["HelpfulnessDenominator"], bins=50, edgecolor="white",
                 alpha=0.8, range=(0, 50))
    axes[1].set_xlabel("Nombre de votes")
    axes[1].set_ylabel("Fréquence")
    axes[1].set_title("Distribution du nombre de votes")
    axes[1].axvline(x=config.MIN_VOTES, color="red", linestyle="--",
                    label=f"Seuil = {config.MIN_VOTES}")
    axes[1].legend()

    plt.tight_layout()
    if save:
        save_fig(fig, "helpfulness_distribution")
    return fig


def plot_temporal_distribution(df, save=True):
    """Distribution temporelle des avis."""
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly = df.groupby(df["DateTime"].dt.to_period("M")).size()
    monthly.index = monthly.index.to_timestamp()
    ax.plot(monthly.index, monthly.values, linewidth=1)
    ax.fill_between(monthly.index, monthly.values, alpha=0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Nombre d'avis")
    ax.set_title("Évolution temporelle des avis")

    import datetime
    split_date = datetime.datetime(config.TRAIN_END_YEAR, 12, 31)
    ax.axvline(x=split_date, color="red", linestyle="--",
               label=f"Découpe train/test ({config.TRAIN_END_YEAR})")
    ax.legend()

    if save:
        save_fig(fig, "temporal_distribution")
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Matrice de confusion",
                          save=True, filename="confusion_matrix"):
    """Affiche la matrice de confusion."""
    from sklearn.metrics import confusion_matrix as cm_func

    cm = cm_func(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels or "auto", yticklabels=labels or "auto")
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(title)
    if save:
        save_fig(fig, filename)
    return fig


def plot_pr_curve(y_true, y_proba, model_name="Model", save=True):
    """Courbe Precision-Recall."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=f"{model_name} (AP={ap:.3f})")
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_title("Courbe Precision-Recall")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    if save:
        save_fig(fig, f"pr_curve_{model_name}")
    return fig


def plot_model_comparison(results_df, metric="f1_macro", save=True):
    """Comparaison des modèles en barres."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_df = results_df.sort_values(metric, ascending=True)
    bars = ax.barh(sorted_df["model"], sorted_df[metric])
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title("Comparaison des modèles")
    for bar, val in zip(bars, sorted_df[metric]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2.,
                f"{val:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    if save:
        save_fig(fig, "model_comparison")
    return fig


def plot_aspect_summary(aspect_summary, save=True):
    """Résumé visuel des aspects."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    aspect_summary["count"].plot(kind="barh", ax=axes[0])
    axes[0].set_xlabel("Nombre de mentions")
    axes[0].set_title("Fréquence des aspects")

    aspect_summary[["pct_negative", "pct_positive"]].plot(
        kind="barh", ax=axes[1], color=["#e74c3c", "#2ecc71"]
    )
    axes[1].set_xlabel("Proportion")
    axes[1].set_title("Sentiment par aspect")
    axes[1].legend(["Négatif", "Positif"])

    plt.tight_layout()
    if save:
        save_fig(fig, "aspect_summary")
    return fig
