"""
charts.py
=========
Composants graphiques réutilisables pour le Dashboard Dash.
Chaque fonction retourne une figure Plotly prête à être intégrée dans dcc.Graph().
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List

GRAVITE_COLORS  = {"Léger": "#27ae60", "Grave": "#f39c12", "Mortel": "#e74c3c"}
GRAVITE_ORDER   = ["Léger", "Grave", "Mortel"]
PALETTE         = {"primary": "#2c3e50", "secondary": "#3498db", "warning": "#f39c12", "danger": "#e74c3c"}

LAYOUT_DEFAULTS = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Segoe UI, Arial", size=11, color="#2c3e50"),
    margin=dict(t=45, b=30, l=40, r=20),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Segoe UI"),
)


def apply_layout(fig: go.Figure, height: int = 320, **kwargs) -> go.Figure:
    """Applique le layout par défaut à une figure."""
    fig.update_layout(height=height, **LAYOUT_DEFAULTS, **kwargs)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# VUE D'ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

def donut_gravite(df: pd.DataFrame) -> go.Figure:
    """Graphique en anneau de la distribution des gravités."""
    counts = df["gravite"].value_counts().reindex(GRAVITE_ORDER, fill_value=0)
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.6,
        marker=dict(
            colors=[GRAVITE_COLORS[g] for g in GRAVITE_ORDER],
            line=dict(color="white", width=2.5)
        ),
        textinfo="percent+label",
        textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>%{value:,} accidents<br>%{percent}<extra></extra>"
    ))
    fig.add_annotation(
        text=f"<b>{len(df):,}</b><br><span style='font-size:10px'>accidents</span>",
        x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#2c3e50")
    )
    return apply_layout(fig, height=300, title="Gravité des accidents", showlegend=False)


def bar_accidents_region(df: pd.DataFrame) -> go.Figure:
    """Barres horizontales : accidents par région (couleur = taux mortalité)."""
    region_data = (
        df.groupby("region")
        .agg(accidents=("accident_id", "count"), deces=("nombre_deces", "sum"))
        .reset_index()
    )
    region_data["taux_mort"] = (region_data["deces"] / region_data["accidents"] * 100).round(1)
    region_data = region_data.sort_values("accidents", ascending=True)

    fig = px.bar(
        region_data, x="accidents", y="region", orientation="h",
        color="taux_mort", color_continuous_scale="YlOrRd",
        text="accidents",
        labels={"accidents": "Nb accidents", "region": "", "taux_mort": "Taux mort. %"},
        custom_data=["taux_mort", "deces"]
    )
    fig.update_traces(
        texttemplate="%{text:,}", textposition="outside",
        hovertemplate="<b>%{y}</b><br>Accidents : %{x:,}<br>Décès : %{customdata[1]:,}<br>Taux mortalité : %{customdata[0]:.1f}%<extra></extra>"
    )
    return apply_layout(fig, height=300, title="Accidents par région (couleur = mortalité)",
                        coloraxis_showscale=False)


def area_tendance_mensuelle(df: pd.DataFrame) -> go.Figure:
    """Aires empilées : évolution mensuelle par gravité."""
    df_copy = df.copy()
    df_copy["periode"] = pd.to_datetime(df_copy["date_accident"]).dt.to_period("M").astype(str)

    monthly = (
        df_copy.groupby(["periode", "gravite"])
        .size().reset_index(name="count")
    )
    # Pivoter pour empilage
    pivot = monthly.pivot(index="periode", columns="gravite", values="count").fillna(0)
    pivot = pivot.reindex(columns=GRAVITE_ORDER, fill_value=0)

    fig = go.Figure()
    for grav in GRAVITE_ORDER:
        if grav in pivot.columns:
            fig.add_trace(go.Scatter(
                x=pivot.index, y=pivot[grav],
                mode="lines", name=grav,
                stackgroup="one", fill="tonexty",
                line=dict(color=GRAVITE_COLORS[grav], width=1.5),
                fillcolor=GRAVITE_COLORS[grav].replace(")", ", 0.6)").replace("rgb", "rgba") if "rgb" in GRAVITE_COLORS[grav] else GRAVITE_COLORS[grav],
                hovertemplate=f"<b>{grav}</b><br>%{{y:,}} accidents<extra></extra>"
            ))
    fig.update_xaxes(tickangle=-30, nticks=12)
    return apply_layout(fig, height=280, title="Évolution mensuelle par gravité",
                        legend=dict(orientation="h", y=1.05, x=0))


def heatmap_heure_jour(df: pd.DataFrame) -> go.Figure:
    """Heatmap : accidents par heure × jour."""
    JOURS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    JOURS_FR = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

    pivot = (
        df.groupby(["heure", "jour_semaine"])
        .size().unstack("jour_semaine", fill_value=0)
        .reindex(columns=JOURS, fill_value=0)
    )
    pivot.columns = JOURS_FR

    fig = px.imshow(
        pivot.T,
        color_continuous_scale="YlOrRd",
        labels={"x": "Heure", "y": "Jour", "color": "Accidents"},
        aspect="auto"
    )
    fig.update_traces(
        hovertemplate="<b>%{y} - %{x}h</b><br>%{z:,} accidents<extra></extra>"
    )
    fig.update_xaxes(title="Heure de la journée")
    fig.update_yaxes(title="")
    return apply_layout(fig, height=300, title="Distribution horaire des accidents",
                        coloraxis_showscale=True)


# ─────────────────────────────────────────────────────────────────────────────
# PROFIL HORAIRE
# ─────────────────────────────────────────────────────────────────────────────

def dual_axis_hourly(df: pd.DataFrame) -> go.Figure:
    """Graphique dual-axe : volume + taux de mortalité par heure."""
    hourly = (
        df.groupby("heure")
        .agg(accidents=("accident_id", "count"), deces=("nombre_deces", "sum"))
        .reset_index()
    )
    hourly["taux"] = (hourly["deces"] / hourly["accidents"] * 100).round(2)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=hourly["heure"], y=hourly["accidents"],
        name="Accidents", marker_color=PALETTE["secondary"],
        opacity=0.75,
        hovertemplate="<b>%{x}h</b><br>Accidents : %{y:,}<extra></extra>"
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=hourly["heure"], y=hourly["taux"],
        name="Taux mortalité (%)", mode="lines+markers",
        line=dict(color=PALETTE["danger"], width=2.5),
        marker=dict(size=6),
        hovertemplate="<b>%{x}h</b><br>Taux : %{y:.1f}%<extra></extra>"
    ), secondary_y=True)

    # Zones nuit
    for span in [(-0.5, 5.5), (21.5, 23.5)]:
        fig.add_vrect(x0=span[0], x1=span[1], fillcolor="navy", opacity=0.06, line_width=0)

    fig.update_xaxes(title="Heure", tickvals=list(range(0, 24, 2)))
    fig.update_yaxes(title="Nombre d'accidents", secondary_y=False)
    fig.update_yaxes(title="Taux mortalité (%)", secondary_y=True, ticksuffix="%")

    return apply_layout(fig, height=340, title="Profil Horaire : Volume & Mortalité",
                        legend=dict(x=0.01, y=0.99))


# ─────────────────────────────────────────────────────────────────────────────
# CAUSES & COMPORTEMENTS
# ─────────────────────────────────────────────────────────────────────────────

def stacked_causes_gravite(df: pd.DataFrame) -> go.Figure:
    """Barres empilées : causes × gravité."""
    cg = (
        df.groupby(["cause", "gravite"])
        .size().reset_index(name="count")
    )
    fig = px.bar(
        cg, x="cause", y="count", color="gravite",
        color_discrete_map=GRAVITE_COLORS,
        category_orders={"gravite": GRAVITE_ORDER},
        barmode="stack",
        labels={"cause": "", "count": "Nb accidents", "gravite": "Gravité"},
        text_auto=False
    )
    fig.update_xaxes(tickangle=-35)
    fig.update_traces(hovertemplate="<b>%{x}</b><br>%{fullData.name} : %{y:,}<extra></extra>")
    return apply_layout(fig, height=340, title="Causes × Gravité",
                        legend=dict(orientation="h", y=1.05))


def scatter_vitesse_cause(df: pd.DataFrame) -> go.Figure:
    """Scatter : cause vs vitesse moyenne (taille = nb accidents)."""
    cause_stats = (
        df.groupby("cause")
        .agg(
            vitesse_moy=("vitesse_estimee", "mean"),
            taux_mort=("gravite", lambda x: (x == "Mortel").mean() * 100),
            n=("accident_id", "count")
        ).reset_index()
    )

    fig = px.scatter(
        cause_stats, x="vitesse_moy", y="taux_mort",
        size="n", color="taux_mort",
        color_continuous_scale="YlOrRd",
        text="cause",
        labels={"vitesse_moy": "Vitesse moyenne (km/h)", "taux_mort": "Taux mortalité (%)"},
        custom_data=["n"]
    )
    fig.update_traces(
        textposition="top center",
        hovertemplate="<b>%{text}</b><br>Vitesse moy : %{x:.1f} km/h<br>Mortalité : %{y:.1f}%<br>Accidents : %{customdata[0]:,}<extra></extra>"
    )
    return apply_layout(fig, height=360, title="Cause : Vitesse vs Taux de mortalité",
                        coloraxis_showscale=False)


def bar_epi_mortalite(df: pd.DataFrame) -> go.Figure:
    """Barres groupées : impact port des EPI sur la mortalité."""
    epi = (
        df.groupby(["port_casque", "port_ceinture"])
        .agg(n=("accident_id", "count"),
             mortel=("gravite", lambda x: (x == "Mortel").sum()))
        .reset_index()
    )
    epi["taux"] = (epi["mortel"] / epi["n"] * 100).round(1)
    epi["label"] = "Casque:" + epi["port_casque"] + " / Ceinture:" + epi["port_ceinture"]
    epi = epi.sort_values("taux", ascending=False)

    colors = [PALETTE["danger"] if t > epi["taux"].mean() else PALETTE["success"]
              for t in epi["taux"]]

    fig = go.Figure(go.Bar(
        x=epi["label"], y=epi["taux"],
        marker_color=colors,
        text=epi["taux"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Taux mortalité : %{y:.1f}%<extra></extra>",
        width=0.6
    ))
    fig.add_hline(y=epi["taux"].mean(), line_dash="dash", line_color="gray",
                  annotation_text=f"Moyenne : {epi['taux'].mean():.1f}%")
    fig.update_xaxes(tickangle=-20)
    return apply_layout(fig, height=320, title="Taux de mortalité selon le port des EPI")


def box_vitesse_gravite(df: pd.DataFrame) -> go.Figure:
    """Boîtes à moustaches : vitesse par gravité."""
    fig = go.Figure()
    for grav in GRAVITE_ORDER:
        sub = df[df["gravite"] == grav]["vitesse_estimee"].dropna()
        fig.add_trace(go.Box(
            y=sub, name=grav,
            marker_color=GRAVITE_COLORS[grav],
            boxmean="sd",
            hovertemplate=f"<b>{grav}</b><br>%{{y:.0f}} km/h<extra></extra>"
        ))
    fig.update_yaxes(title="Vitesse estimée (km/h)")
    return apply_layout(fig, height=320, title="Vitesse estimée par gravité",
                        showlegend=False)


# ─────────────────────────────────────────────────────────────────────────────
# ÉCONOMIQUE
# ─────────────────────────────────────────────────────────────────────────────

def bar_cout_region(df: pd.DataFrame) -> go.Figure:
    """Barres horizontales : coût total par région."""
    cout = (
        df.groupby("region")["cout_estime"]
        .sum().reset_index()
        .sort_values("cout_estime", ascending=True)
    )
    cout["cout_M"] = (cout["cout_estime"] / 1e6).round(1)

    fig = px.bar(
        cout, x="cout_M", y="region", orientation="h",
        color="cout_M", color_continuous_scale="Blues",
        text="cout_M",
        labels={"cout_M": "Coût (MFCFA)", "region": ""},
    )
    fig.update_traces(
        texttemplate="%{text:.1f} M",
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:.1f} MFCFA<extra></extra>"
    )
    return apply_layout(fig, height=300, title="Coût économique total par région (MFCFA)",
                        coloraxis_showscale=False)


def violin_cout_gravite(df: pd.DataFrame) -> go.Figure:
    """Violon : distribution du coût par gravité."""
    fig = go.Figure()
    for grav in GRAVITE_ORDER:
        sub = df[df["gravite"] == grav]["cout_estime"].dropna()
        fig.add_trace(go.Violin(
            y=sub, name=grav,
            fillcolor=GRAVITE_COLORS[grav],
            line_color=GRAVITE_COLORS[grav],
            opacity=0.7, box_visible=True, meanline_visible=True,
            hovertemplate=f"<b>{grav}</b><br>%{{y:,.0f}} FCFA<extra></extra>"
        ))
    fig.update_yaxes(title="Coût estimé (FCFA)")
    return apply_layout(fig, height=340, title="Distribution du coût par gravité",
                        showlegend=False)


def line_evolution_cout(df: pd.DataFrame) -> go.Figure:
    """Ligne : évolution du coût total annuel."""
    annual = (
        df.assign(annee=pd.to_datetime(df["date_accident"]).dt.year)
        .groupby("annee")["cout_estime"]
        .sum().reset_index()
    )
    annual["cout_G"] = annual["cout_estime"] / 1e9

    fig = go.Figure(go.Scatter(
        x=annual["annee"], y=annual["cout_G"],
        mode="lines+markers+text",
        line=dict(color=PALETTE["warning"], width=3),
        marker=dict(size=10, color=PALETTE["warning"]),
        fill="tozeroy", fillcolor="rgba(243,156,18,0.15)",
        text=annual["cout_G"].apply(lambda x: f"{x:.2f}G"),
        textposition="top center",
        hovertemplate="<b>%{x}</b><br>%{y:.3f} GFCFA<extra></extra>"
    ))
    fig.update_xaxes(title="Année", dtick=1)
    fig.update_yaxes(title="Coût total (GFCFA)")
    return apply_layout(fig, height=300, title="Évolution du coût économique annuel (GFCFA)")


# ─────────────────────────────────────────────────────────────────────────────
# CARTOGRAPHIE PLOTLY
# ─────────────────────────────────────────────────────────────────────────────

def scatter_map(df: pd.DataFrame, max_points: int = 6000) -> go.Figure:
    """Carte scatter des accidents (Plotly Mapbox)."""
    sample = df.sample(min(max_points, len(df)), random_state=42)

    fig = px.scatter_mapbox(
        sample,
        lat="latitude", lon="longitude",
        color="gravite",
        color_discrete_map=GRAVITE_COLORS,
        size_max=7, opacity=0.55,
        zoom=6, center={"lat": 8.0, "lon": 1.1},
        mapbox_style="carto-positron",
        hover_data={"ville": True, "cause": True, "type_accident": True,
                    "latitude": False, "longitude": False},
        labels={"gravite": "Gravité"}
    )
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>Cause : %{customdata[1]}<br>Type : %{customdata[2]}<extra></extra>"
    )
    return apply_layout(fig, height=480, title=f"Localisation des accidents ({len(sample):,} points)",
                        margin=dict(t=40, b=0, l=0, r=0),
                        legend=dict(x=0.01, y=0.99))


def density_map(df: pd.DataFrame) -> go.Figure:
    """Carte de densité (heatmap géographique Plotly)."""
    fig = px.density_mapbox(
        df, lat="latitude", lon="longitude",
        radius=8, zoom=6, center={"lat": 8.0, "lon": 1.1},
        mapbox_style="carto-positron",
        color_continuous_scale="YlOrRd",
        opacity=0.7,
    )
    return apply_layout(fig, height=450, title="Densité des accidents (heatmap géographique)",
                        margin=dict(t=40, b=0, l=0, r=0),
                        coloraxis_colorbar=dict(title="Densité"))


# ─────────────────────────────────────────────────────────────────────────────
# RISQUE PAR VÉHICULE
# ─────────────────────────────────────────────────────────────────────────────

def radar_vehicule(df: pd.DataFrame) -> go.Figure:
    """Graphique radar des indicateurs de risque par type de véhicule."""
    types = df["type_vehicule"].value_counts().nlargest(5).index.tolist()
    metrics = {
        "Taux mort. (%)":   lambda g: (g["gravite"] == "Mortel").mean() * 100,
        "Vitesse moy.":     lambda g: g["vitesse_estimee"].mean() / 2,  # Normalisé /2
        "Sans EPI (%)":     lambda g: ((g["port_casque"] == "Non") | (g["port_ceinture"] == "Non")).mean() * 100,
        "Nuit (%)":         lambda g: (g["luminosite"] == "Nuit").mean() * 100,
        "Excès vit. (%)":   lambda g: (g["vitesse_estimee"] > g["limitation_vitesse"]).mean() * 100,
    }
    categories = list(metrics.keys())

    fig = go.Figure()
    colors = px.colors.qualitative.Set2[:len(types)]

    for veh, color in zip(types, colors):
        sub = df[df["type_vehicule"] == veh]
        values = [fn(sub) for fn in metrics.values()]
        values_closed = values + [values[0]]
        cats_closed = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values_closed, theta=cats_closed,
            fill="toself", name=veh,
            line=dict(color=color, width=2),
            fillcolor=color.replace("rgb", "rgba").replace(")", ", 0.15)") if "rgb" in color else color,
            opacity=0.8
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        legend=dict(x=1.05, y=0.5)
    )
    return apply_layout(fig, height=380, title="Profil de risque par type de véhicule")


def treemap_region_gravite(df: pd.DataFrame) -> go.Figure:
    """Treemap : région × gravité."""
    data = (
        df.groupby(["region", "gravite"])
        .size().reset_index(name="count")
    )
    fig = px.treemap(
        data,
        path=[px.Constant("Togo"), "region", "gravite"],
        values="count",
        color="gravite",
        color_discrete_map=GRAVITE_COLORS,
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>%{value:,} accidents<br>%{percentParent:.1%} de la zone parent<extra></extra>"
    )
    return apply_layout(fig, height=380, title="Treemap : Accidents par Région et Gravité")
