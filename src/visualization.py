"""
visualization.py
================
Fonctions de visualisation de niveau expert pour le projet Togo Road Accidents.
Matplotlib / Seaborn / Plotly — style publication-ready.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")

from utils import get_logger, GRAVITE_COLORS, PALETTE

logger = get_logger("visualization")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION GLOBALE
# ─────────────────────────────────────────────────────────────────────────────

FIGSIZE_WIDE  = (16, 6)
FIGSIZE_SQUARE = (10, 8)
FIGSIZE_TALL   = (10, 14)

COLORS_GRAVITE = [GRAVITE_COLORS["Léger"], GRAVITE_COLORS["Grave"], GRAVITE_COLORS["Mortel"]]
CMAP_RISK = "YlOrRd"

def set_style():
    """Configure le style global Matplotlib."""
    plt.rcParams.update({
        "figure.dpi":        150,
        "figure.facecolor":  "white",
        "axes.facecolor":    "#f8f9fa",
        "axes.grid":         True,
        "grid.alpha":        0.4,
        "grid.linestyle":    "--",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    14,
        "axes.labelsize":    12,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
        "figure.titlesize":  16,
    })

set_style()


def save_fig(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    """Sauvegarde un graphique Matplotlib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Figure sauvegardée : {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_gravite_distribution(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Graphique en anneau + barres pour la distribution de la gravité.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    fig.suptitle("Distribution des Accidents par Gravité — Togo 2022-2025", fontsize=16, fontweight="bold")

    counts = df["gravite"].value_counts().reindex(["Léger", "Grave", "Mortel"])

    # Donut
    ax1 = axes[0]
    wedges, texts, autotexts = ax1.pie(
        counts,
        labels=counts.index,
        colors=COLORS_GRAVITE,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.8,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax1.set_title("Répartition (%) par gravité", pad=20)

    # Barres horizontales avec annotations
    ax2 = axes[1]
    bars = ax2.barh(counts.index, counts.values, color=COLORS_GRAVITE,
                    edgecolor="white", linewidth=1.5, height=0.6)
    for bar, val in zip(bars, counts.values):
        ax2.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
                 f"{val:,}", va="center", ha="left", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Nombre d'accidents")
    ax2.set_title("Volume par niveau de gravité")
    ax2.set_xlim(0, counts.max() * 1.12)

    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_temporal_heatmap(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Heatmap heure × jour de la semaine pour le nombre d'accidents.
    """
    JOURS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    JOURS_FR = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

    pivot = (
        df.groupby(["heure", "jour_semaine"])
        .size()
        .unstack("jour_semaine")
        .reindex(columns=JOURS)
    )
    pivot.columns = JOURS_FR

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd",
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Nombre d'accidents", "shrink": 0.8},
        annot=True,           # ✅ affiche les nombres
        fmt="d",              # ✅ format entier (pas de décimales)
        annot_kws={"size": 7} # ✅ taille réduite pour éviter le chevauchement
    )
    ax.set_title("Heatmap Temporelle — Accidents par Heure et Jour de Semaine",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Jour de la semaine", fontsize=12)
    ax.set_ylabel("Heure de la journée", fontsize=12)
    ax.tick_params(axis="x", rotation=45)

    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_hourly_risk(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Profil horaire : volume d'accidents + taux de mortalité.
    """
    hourly = df.groupby("heure").agg(
        nb_accidents=("accident_id", "count"),
        deces=("nombre_deces", "sum")
    ).reset_index()
    hourly["taux_mortalite"] = hourly["deces"] / hourly["nb_accidents"] * 100

    fig, ax1 = plt.subplots(figsize=FIGSIZE_WIDE)
    fig.suptitle("Profil Horaire des Accidents et Taux de Mortalité", fontsize=15, fontweight="bold")

    # Barres : volume
    bars = ax1.bar(hourly["heure"], hourly["nb_accidents"],
                   color=PALETTE["secondary"], alpha=0.7, label="Nombre d'accidents")
    ax1.set_xlabel("Heure de la journée", fontsize=12)
    ax1.set_ylabel("Nombre d'accidents", color=PALETTE["secondary"], fontsize=12)
    ax1.tick_params(axis="y", labelcolor=PALETTE["secondary"])
    ax1.set_xticks(range(24))

    # Ligne : taux de mortalité
    ax2 = ax1.twinx()
    ax2.plot(hourly["heure"], hourly["taux_mortalite"],
             color=PALETTE["danger"], linewidth=2.5, marker="o",
             markersize=5, label="Taux mortalité (%)")
    ax2.set_ylabel("Taux de mortalité (%)", color=PALETTE["danger"], fontsize=12)
    ax2.tick_params(axis="y", labelcolor=PALETTE["danger"])

    # Zones colorées : nuit profonde
    ax1.axvspan(-0.5, 5.5, alpha=0.08, color="navy", label="Nuit profonde")
    ax1.axvspan(21.5, 23.5, alpha=0.08, color="navy")

    # Légende combinée
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    if save_path:
        save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. ANALYSES COMPARATIVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_causes_analysis(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Analyse des causes : fréquence + taux de mortalité.
    """
    causes = (
        df.groupby("cause")
        .agg(nb=("accident_id", "count"), mortel=("gravite", lambda x: (x == "Mortel").sum()))
        .reset_index()
    )
    causes["taux_mortalite"] = causes["mortel"] / causes["nb"] * 100
    causes = causes.sort_values("nb", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    fig.suptitle("Analyse des Causes d'Accidents", fontsize=15, fontweight="bold")

    # Barres horizontales : fréquence
    colors_freq = plt.cm.Blues(np.linspace(0.4, 0.9, len(causes)))
    axes[0].barh(causes["cause"], causes["nb"], color=colors_freq, edgecolor="white")
    for i, (_, row) in enumerate(causes.iterrows()):
        axes[0].text(row["nb"] + 50, i, f"{row['nb']:,}", va="center", fontsize=9)
    axes[0].set_title("Fréquence par cause", fontsize=12)
    axes[0].set_xlabel("Nombre d'accidents")

    # Barres horizontales : taux de mortalité
    colors_mort = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(causes)))
    axes[1].barh(causes["cause"], causes["taux_mortalite"], color=colors_mort, edgecolor="white")
    for i, (_, row) in enumerate(causes.iterrows()):
        axes[1].text(row["taux_mortalite"] + 0.1, i, f"{row['taux_mortalite']:.1f}%", va="center", fontsize=9)
    axes[1].set_title("Taux de mortalité par cause (%)", fontsize=12)
    axes[1].set_xlabel("Taux de mortalité (%)")

    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_speed_boxplot(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Distribution de la vitesse estimée par gravité et type de route.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    fig.suptitle("Distribution de la Vitesse selon la Gravité et le Type de Route",
                 fontsize=14, fontweight="bold")

    # Boxplot vitesse × gravité
    order = ["Léger", "Grave", "Mortel"]
    sns.boxplot(
        data=df, x="gravite", y="vitesse_estimee", order=order,
        palette=GRAVITE_COLORS, ax=axes[0], linewidth=1.5,
        flierprops=dict(marker="o", alpha=0.3, markersize=3)
    )
    axes[0].set_title("Vitesse par niveau de gravité")
    axes[0].set_xlabel("Gravité")
    axes[0].set_ylabel("Vitesse estimée (km/h)")

    # Violin vitesse × type de route
    sns.violinplot(
        data=df, x="type_route", y="vitesse_estimee",
        palette="Set2", ax=axes[1], linewidth=1, inner="quartile"
    )
    axes[1].set_title("Vitesse par type de route")
    axes[1].set_xlabel("Type de route")
    axes[1].set_ylabel("Vitesse estimée (km/h)")

    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_region_radar(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Graphique radar des indicateurs de risque par région.
    """
    regions = df["region"].unique()
    metrics = {
        "Taux mortalité (%)":   lambda g: (g["gravite"] == "Mortel").mean() * 100,
        "Vitesse moy. (km/h)":  lambda g: g["vitesse_estimee"].mean(),
        "Nuit (%)":             lambda g: (g["luminosite"] == "Nuit").mean() * 100,
        "Sans EPI (%)":         lambda g: ((g["port_casque"] == "Non") | (g["port_ceinture"] == "Non")).mean() * 100,
        "Pluie (%)":            lambda g: (g["meteo"] == "Pluie").mean() * 100,
    }

    region_stats = {}
    for region in regions:
        sub = df[df["region"] == region]
        region_stats[region] = {m: fn(sub) for m, fn in metrics.items()}

    stats_df = pd.DataFrame(region_stats).T
    # Normaliser chaque métrique entre 0 et 1
    stats_norm = (stats_df - stats_df.min()) / (stats_df.max() - stats_df.min() + 1e-10)

    categories = list(metrics.keys())
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(regions)))

    for (region, row), color in zip(stats_norm.iterrows(), colors):
        values = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, values, "o-", linewidth=2, label=region, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_yticklabels([])
    ax.set_title("Profil de Risque par Région (normalisé)", size=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.grid(True, alpha=0.4)

    if save_path:
        save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. ANALYSES TEMPORELLES
# ─────────────────────────────────────────────────────────────────────────────

def plot_tendance_annuelle(df: pd.DataFrame, save_path: Optional[Path] = None) -> go.Figure:
    """
    Évolution mensuelle interactive (Plotly) des accidents et décès.
    """
    monthly = (
        df.assign(periode=df["date_accident"].dt.to_period("M"))
        .groupby("periode")
        .agg(accidents=("accident_id", "count"), deces=("nombre_deces", "sum"))
        .reset_index()
    )
    monthly["periode"] = monthly["periode"].astype(str)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Nombre d'accidents par mois", "Nombre de décès par mois"),
                        vertical_spacing=0.08)

    fig.add_trace(go.Scatter(
        x=monthly["periode"], y=monthly["accidents"],
        fill="tozeroy", fillcolor="rgba(52,152,219,0.2)",
        line=dict(color="#3498db", width=2), name="Accidents"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=monthly["periode"], y=monthly["deces"],
        fill="tozeroy", fillcolor="rgba(231,76,60,0.2)",
        line=dict(color="#e74c3c", width=2), name="Décès"
    ), row=2, col=1)

    fig.update_layout(
        title="Évolution Mensuelle des Accidents de la Route au Togo (2022-2025)",
        height=500, showlegend=True,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=12)
    )
    return fig


def plot_sankey_causes_gravite(df: pd.DataFrame) -> go.Figure:
    """
    Diagramme de Sankey : Cause → Type d'accident → Gravité.
    """
    causes = df["cause"].value_counts().nlargest(6).index.tolist()
    sub = df[df["cause"].isin(causes)]

    # Construire les nœuds et liens
    nodes_cause = causes
    nodes_gravite = ["Léger", "Grave", "Mortel"]
    all_nodes = nodes_cause + nodes_gravite

    links_source, links_target, links_value, links_color = [], [], [], []
    color_map = {"Léger": "rgba(46,204,113,0.4)", "Grave": "rgba(243,156,18,0.4)", "Mortel": "rgba(231,76,60,0.4)"}

    for i, cause in enumerate(nodes_cause):
        for j, grav in enumerate(nodes_gravite):
            count = len(sub[(sub["cause"] == cause) & (sub["gravite"] == grav)])
            if count > 0:
                links_source.append(i)
                links_target.append(len(nodes_cause) + j)
                links_value.append(count)
                links_color.append(color_map[grav])

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=all_nodes,
            color=["rgba(52,152,219,0.8)"] * len(nodes_cause) +
                  ["rgba(46,204,113,0.8)", "rgba(243,156,18,0.8)", "rgba(231,76,60,0.8)"]
        ),
        link=dict(source=links_source, target=links_target, value=links_value, color=links_color)
    ))
    fig.update_layout(
        title="Flux Cause → Gravité des Accidents",
        font=dict(size=11), height=450,
        plot_bgcolor="white", paper_bgcolor="white"
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. TABLEAU DE BORD SYNTHÉTIQUE
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard_overview(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Vue d'ensemble en un seul graphique (4 panels).
    """
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    fig.suptitle("Vue d'Ensemble — Accidents de la Route au Togo 2022-2025",
                 fontsize=18, fontweight="bold", y=1.01)

    # Panel 1 : Gravité
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df["gravite"].value_counts().reindex(["Léger", "Grave", "Mortel"])
    wedges, _, autotexts = ax1.pie(
        counts, labels=counts.index, colors=COLORS_GRAVITE,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2)
    )
    ax1.set_title("Répartition par gravité", fontweight="bold")

    # Panel 2 : Accidents par région
    ax2 = fig.add_subplot(gs[0, 1])
    region_counts = df["region"].value_counts()
    bars = ax2.bar(region_counts.index, region_counts.values,
                   color=plt.cm.Blues(np.linspace(0.4, 0.85, len(region_counts))),
                   edgecolor="white")
    ax2.set_title("Accidents par région", fontweight="bold")
    ax2.set_ylabel("Nombre d'accidents")
    ax2.tick_params(axis="x", rotation=30)

    # Panel 3 : Top causes
    ax3 = fig.add_subplot(gs[0, 2])
    causes = df["cause"].value_counts().nlargest(6)
    ax3.barh(causes.index[::-1], causes.values[::-1],
             color=plt.cm.Oranges(np.linspace(0.4, 0.9, len(causes))),
             edgecolor="white")
    ax3.set_title("Top causes d'accidents", fontweight="bold")
    ax3.set_xlabel("Nombre d'accidents")

    # Panel 4 : Heatmap heure × période journalière
    ax4 = fig.add_subplot(gs[1, :2])
    pivot_h = df.groupby(["heure", "periode_journaliere"]).size().unstack(fill_value=0)
    sns.heatmap(pivot_h.T, ax=ax4, cmap="YlOrRd", linewidths=0.2,
                cbar_kws={"shrink": 0.8, "label": "Nb accidents"})
    ax4.set_title("Distribution horaire par période", fontweight="bold")
    ax4.set_xlabel("Heure")

    # Panel 5 : Vitesse par gravité
    ax5 = fig.add_subplot(gs[1, 2])
    for i, (grav, color) in enumerate(GRAVITE_COLORS.items()):
        sub = df[df["gravite"] == grav]["vitesse_estimee"].dropna()
        ax5.violinplot([sub.values], positions=[i], widths=0.7,
                       showmedians=True, showextrema=False)
    ax5.set_xticks([0, 1, 2])
    ax5.set_xticklabels(["Léger", "Grave", "Mortel"])
    ax5.set_title("Vitesse estimée par gravité", fontweight="bold")
    ax5.set_ylabel("Vitesse (km/h)")

    if save_path:
        save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISATIONS ML
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, classes: List[str],
                           save_path: Optional[Path] = None) -> plt.Figure:
    """Matrice de confusion normalisée et brute côte à côte."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Matrices de Confusion", fontsize=14, fontweight="bold")

    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    for ax, data, title, fmt in [
        (axes[0], cm, "Absolue", "d"),
        (axes[1], cm_norm, "Normalisée (recall)", ".2f")
    ]:
        im = ax.imshow(data, interpolation="nearest", cmap="Blues")
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ticks = np.arange(len(classes))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        thresh = data.max() / 2.0
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{data[i, j]:{fmt}}",
                        ha="center", va="center", fontsize=12,
                        color="white" if data[i, j] > thresh else "black")
        ax.set_ylabel("Vrai label")
        ax.set_xlabel("Prédit")

    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_feature_importance(importances: pd.Series, top_n: int = 20,
                             title: str = "Feature Importances",
                             save_path: Optional[Path] = None) -> plt.Figure:
    """Graphique des importances de features (horizontal bars)."""
    top = importances.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(10, top_n * 0.4 + 2))

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top)))
    bars = ax.barh(top.index, top.values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, top.values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.set_xlim(0, top.max() * 1.15)

    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_roc_curves(roc_data: dict, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Courbes ROC OvR pour classification multiclasse.
    roc_data: dict {class_name: {"fpr": array, "tpr": array, "auc": float}}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [GRAVITE_COLORS["Léger"], GRAVITE_COLORS["Grave"], GRAVITE_COLORS["Mortel"]]

    for (cls, data), color in zip(roc_data.items(), colors):
        ax.plot(data["fpr"], data["tpr"], color=color, linewidth=2.5,
                label=f"{cls} (AUC = {data['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Aléatoire")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")

    ax.set_xlabel("Taux de Faux Positifs", fontsize=12)
    ax.set_ylabel("Taux de Vrais Positifs", fontsize=12)
    ax.set_title("Courbes ROC — Classification de la Gravité (OvR)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    if save_path:
        save_fig(fig, save_path)
    return fig


if __name__ == "__main__":
    from utils import CLEAN_DATA_FILE, PATHS, load_dataframe
    df = load_dataframe(CLEAN_DATA_FILE)

    fig1 = plot_gravite_distribution(df, PATHS["figures"] / "01_gravite_distribution.png")
    fig2 = plot_temporal_heatmap(df, PATHS["figures"] / "02_heatmap_temporelle.png")
    fig3 = plot_hourly_risk(df, PATHS["figures"] / "03_profil_horaire.png")
    fig4 = plot_causes_analysis(df, PATHS["figures"] / "04_causes.png")
    fig5 = plot_dashboard_overview(df, PATHS["figures"] / "05_dashboard.png")

    plt.show()
    logger.info("Toutes les visualisations générées.")
