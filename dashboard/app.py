"""
app.py
======
Dashboard interactif Dash pour l'analyse des accidents de la route au Togo.
Lancer avec : python app.py
Accès       : http://localhost:8050
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "accidents_clean.parquet"

def load_data() -> pd.DataFrame:
    """Charge les données nettoyées."""
    if DATA_PATH.exists():
        return pd.read_parquet(DATA_PATH)
    else:
        # Fallback sur le CSV brut
        raw_path = Path(__file__).parent.parent / "data" / "raw" / "togo_road_accidents_dataset.csv"
        df = pd.read_csv(raw_path, parse_dates=["date_accident"])
        # Nettoyage minimal
        df["heure"] = df["heure"] % 24
        df["vitesse_estimee"] = df["vitesse_estimee"].fillna(df["vitesse_estimee"].median())
        return df

df = load_data()

# Variables globales pour les filtres
REGIONS    = sorted(df["region"].unique().tolist())
GRAVITES   = ["Léger", "Grave", "Mortel"]
ANNEES     = sorted(df["date_accident"].dt.year.unique().tolist())
TYPES_VEH  = sorted(df["type_vehicule"].unique().tolist())

GRAVITE_COLORS = {"Léger": "#27ae60", "Grave": "#f39c12", "Mortel": "#e74c3c"}
THEME_COLOR    = "#2c3e50"
CARD_STYLE     = {"borderRadius": "10px", "boxShadow": "0 2px 10px rgba(0,0,0,0.08)", "border": "none"}


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION DASH
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Togo Road Safety Dashboard"
)

server = app.server  # Pour déploiement Gunicorn/Railway


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSANTS RÉUTILISABLES
# ─────────────────────────────────────────────────────────────────────────────

def make_kpi_card(title: str, value: str, icon: str, color: str, subtitle: str = "") -> dbc.Card:
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.H2(value, className="fw-bold mb-0", style={"color": color, "fontSize": "2rem"}),
                    html.P(title, className="text-muted mb-0", style={"fontSize": "0.85rem"}),
                    html.Small(subtitle, className="text-muted") if subtitle else html.Span(),
                ], style={"flex": "1"}),
                html.Div([
                    html.I(className=f"bi bi-{icon}", style={"fontSize": "2.5rem", "color": color, "opacity": "0.3"})
                ])
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"})
        ])
    ], style=CARD_STYLE, className="h-100")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTRES
# ─────────────────────────────────────────────────────────────────────────────

sidebar = dbc.Card([
    dbc.CardHeader([
        html.I(className="bi bi-funnel-fill me-2"),
        html.Strong("Filtres"),
    ], className="fw-bold"),
    dbc.CardBody([
        html.Label("Années", className="fw-semibold small"),
        dcc.RangeSlider(
            id="filter-annee",
            min=min(ANNEES), max=max(ANNEES),
            step=1, value=[min(ANNEES), max(ANNEES)],
            marks={y: str(y) for y in ANNEES},
            tooltip={"placement": "bottom"}
        ),
        html.Hr(),

        html.Label("Régions", className="fw-semibold small"),
        dcc.Dropdown(
            id="filter-region",
            options=[{"label": r, "value": r} for r in REGIONS],
            value=None, multi=True, placeholder="Toutes les régions",
            clearable=True
        ),
        html.Hr(),

        html.Label("Gravité", className="fw-semibold small"),
        dcc.Checklist(
            id="filter-gravite",
            options=[{"label": f"  {g}", "value": g} for g in GRAVITES],
            value=GRAVITES,
            inputStyle={"marginRight": "8px"},
            labelStyle={"display": "flex", "alignItems": "center", "marginBottom": "4px"}
        ),
        html.Hr(),

        html.Label("Type de véhicule", className="fw-semibold small"),
        dcc.Dropdown(
            id="filter-vehicule",
            options=[{"label": v, "value": v} for v in TYPES_VEH],
            value=None, multi=True, placeholder="Tous les véhicules",
            clearable=True
        ),
        html.Hr(),

        html.Label("Période de la journée", className="fw-semibold small"),
        dcc.Dropdown(
            id="filter-periode",
            options=[{"label": p, "value": p} for p in ["Matin", "Après-midi", "Soir", "Nuit"]],
            value=None, multi=True, placeholder="Toutes les périodes",
            clearable=True
        ),
        html.Hr(),

        dbc.Button(
            [html.I(className="bi bi-arrow-counterclockwise me-1"), "Réinitialiser"],
            id="btn-reset", color="secondary", outline=True, size="sm", className="w-100"
        ),
    ])
], style=CARD_STYLE)


# ─────────────────────────────────────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────────────────────────────────────

tabs = dbc.Tabs([
    dbc.Tab(label="📊 Vue d'ensemble",    tab_id="tab-overview"),
    dbc.Tab(label="⏰ Analyse temporelle", tab_id="tab-temporal"),
    dbc.Tab(label="🗺️ Cartographie",      tab_id="tab-geo"),
    dbc.Tab(label="⚠️ Causes & Risques",  tab_id="tab-causes"),
    dbc.Tab(label="💰 Économique",         tab_id="tab-eco"),
    dbc.Tab(label="🤖 Prédiction ML",      tab_id="tab-ml"),
], id="main-tabs", active_tab="tab-overview", className="mb-3")


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H2([
                    html.I(className="bi bi-shield-exclamation me-2", style={"color": "#e74c3c"}),
                    "Togo Road Safety Analytics"
                ], className="mb-0 fw-bold", style={"color": THEME_COLOR}),
                html.P("Tableau de bord d'analyse des accidents de la route 2022-2025",
                       className="text-muted mb-0")
            ])
        ], md=8),
        dbc.Col([
            html.Div([
                html.Span("Données actualisées : 2022-2025", className="badge bg-success me-2"),
                html.Span("50 500 accidents", className="badge bg-primary"),
            ], className="d-flex justify-content-end align-items-center h-100")
        ], md=4)
    ], className="py-3 mb-2 border-bottom"),

    # KPIs (toujours visibles)
    dbc.Row(id="kpi-row", className="mb-3 g-3"),

    # Corps principal
    dbc.Row([
        dbc.Col(sidebar, md=2),
        dbc.Col([
            tabs,
            html.Div(id="tab-content")
        ], md=10)
    ]),

    # Store pour les données filtrées
    dcc.Store(id="filtered-data-store"),

], fluid=True, className="px-4 py-2")


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("filtered-data-store", "data"),
    [Input("filter-annee", "value"),
     Input("filter-region", "value"),
     Input("filter-gravite", "value"),
     Input("filter-vehicule", "value"),
     Input("filter-periode", "value")]
)
def filter_data(annees, regions, gravites, vehicules, periodes):
    """Applique les filtres et stocke les données filtrées."""
    filtered = df.copy()

    # Filtre années
    if annees:
        filtered = filtered[
            (filtered["date_accident"].dt.year >= annees[0]) &
            (filtered["date_accident"].dt.year <= annees[1])
        ]
    # Filtre régions
    if regions:
        filtered = filtered[filtered["region"].isin(regions)]
    # Filtre gravité
    if gravites:
        filtered = filtered[filtered["gravite"].isin(gravites)]
    # Filtre véhicule
    if vehicules:
        filtered = filtered[filtered["type_vehicule"].isin(vehicules)]
    # Filtre période
    if periodes:
        filtered = filtered[filtered["periode_journaliere"].isin(periodes)]

    # Retourner les données sérialisées (sample pour performance)
    return filtered.to_dict("records")


@app.callback(
    Output("kpi-row", "children"),
    Input("filtered-data-store", "data")
)
def update_kpis(data):
    """Met à jour les KPIs."""
    if not data:
        return []
    filtered = pd.DataFrame(data)

    total    = len(filtered)
    deces    = int(filtered["nombre_deces"].sum())
    victimes = int(filtered["nombre_victimes"].sum())
    tx_mort  = deces / total * 100 if total > 0 else 0
    cout     = filtered["cout_estime"].sum() / 1e9

    kpis = [
        make_kpi_card("Total accidents", f"{total:,}", "car-front", "#3498db"),
        make_kpi_card("Décès", f"{deces:,}", "heart-pulse", "#e74c3c", f"Taux: {tx_mort:.1f}%"),
        make_kpi_card("Victimes", f"{victimes:,}", "people-fill", "#f39c12"),
        make_kpi_card("Coût estimé", f"{cout:.1f} GFCFA", "cash-stack", "#27ae60"),
        make_kpi_card("Régions couvertes",
                       str(filtered["region"].nunique()), "geo-alt", "#9b59b6"),
    ]
    return [dbc.Col(card, md=2) for card in kpis]


@app.callback(
    Output("tab-content", "children"),
    [Input("main-tabs", "active_tab"),
     Input("filtered-data-store", "data")]
)
def render_tab(active_tab, data):
    """Rend le contenu de l'onglet actif."""
    if not data:
        return dbc.Alert("Aucune donnée disponible avec les filtres sélectionnés.", color="warning")

    filtered = pd.DataFrame(data)

    if active_tab == "tab-overview":
        return render_overview(filtered)
    elif active_tab == "tab-temporal":
        return render_temporal(filtered)
    elif active_tab == "tab-geo":
        return render_geo(filtered)
    elif active_tab == "tab-causes":
        return render_causes(filtered)
    elif active_tab == "tab-eco":
        return render_economic(filtered)
    elif active_tab == "tab-ml":
        return render_ml_predictor()
    return html.Div("Onglet inconnu")


# ─────────────────────────────────────────────────────────────────────────────
# RENDUS DES ONGLETS
# ─────────────────────────────────────────────────────────────────────────────

def render_overview(filtered: pd.DataFrame) -> html.Div:
    """Onglet Vue d'ensemble."""
    # Graphique 1 : Donut gravité
    counts = filtered["gravite"].value_counts().reindex(GRAVITES, fill_value=0)
    fig_donut = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        hole=0.55,
        marker=dict(colors=[GRAVITE_COLORS[g] for g in GRAVITES], line=dict(color="white", width=2)),
        textinfo="percent+label"
    ))
    fig_donut.update_layout(
        title="Répartition par gravité", showlegend=False, height=300,
        margin=dict(t=40, b=10, l=10, r=10)
    )

    # Graphique 2 : Accidents par région
    region_data = (
        filtered.groupby("region")
        .agg(accidents=("accident_id", "count"), deces=("nombre_deces", "sum"))
        .reset_index()
    )
    region_data["taux_mort"] = region_data["deces"] / region_data["accidents"] * 100
    fig_region = px.bar(
        region_data.sort_values("accidents", ascending=True),
        x="accidents", y="region", orientation="h",
        color="taux_mort", color_continuous_scale="YlOrRd",
        labels={"accidents": "Nb accidents", "region": "Région", "taux_mort": "Taux mort. %"},
        title="Accidents par région (couleur = taux mortalité)"
    )
    fig_region.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))

    # Graphique 3 : Top types d'accidents
    type_data = filtered["type_accident"].value_counts().nlargest(6).reset_index()
    type_data.columns = ["type", "count"]
    fig_types = px.bar(
        type_data, x="count", y="type", orientation="h",
        color="count", color_continuous_scale="Blues",
        title="Types d'accidents les plus fréquents"
    )
    fig_types.update_layout(height=300, showlegend=False, margin=dict(t=40, b=10, l=10, r=10))

    # Graphique 4 : Évolution mensuelle
    monthly = (
        filtered.assign(mois=pd.to_datetime(filtered["date_accident"]).dt.to_period("M"))
        .groupby("mois").size().reset_index()
    )
    monthly["mois"] = monthly["mois"].astype(str)
    monthly.columns = ["mois", "accidents"]
    fig_trend = px.area(
        monthly, x="mois", y="accidents",
        title="Évolution mensuelle des accidents",
        labels={"mois": "Mois", "accidents": "Nb accidents"},
        color_discrete_sequence=["#3498db"]
    )
    fig_trend.update_layout(height=250, margin=dict(t=40, b=10, l=10, r=10))

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_donut), md=4),
            dbc.Col(dcc.Graph(figure=fig_region), md=4),
            dbc.Col(dcc.Graph(figure=fig_types), md=4),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_trend), md=12),
        ])
    ])


def render_temporal(filtered: pd.DataFrame) -> html.Div:
    """Onglet Analyse temporelle."""
    filtered = filtered.copy()
    filtered["date_accident"] = pd.to_datetime(filtered["date_accident"])

    # Heatmap heure × jour
    JOURS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    JOURS_FR = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    pivot = (
        filtered.groupby(["heure", "jour_semaine"])
        .size().unstack("jour_semaine", fill_value=0)
        .reindex(columns=JOURS_ORDER, fill_value=0)
    )
    pivot.columns = JOURS_FR
    fig_heatmap = px.imshow(
        pivot.T, color_continuous_scale="YlOrRd",
        title="Heatmap : Accidents par heure et jour de la semaine",
        labels={"x": "Heure", "y": "Jour", "color": "Nb accidents"},
        aspect="auto"
    )
    fig_heatmap.update_layout(height=350)

    # Profil horaire avec taux de mortalité
    hourly = (
        filtered.groupby("heure")
        .agg(accidents=("accident_id", "count"), deces=("nombre_deces", "sum"))
        .reset_index()
    )
    hourly["taux_mortalite"] = hourly["deces"] / hourly["accidents"] * 100

    fig_hourly = make_subplots(specs=[[{"secondary_y": True}]])
    fig_hourly.add_trace(
        go.Bar(x=hourly["heure"], y=hourly["accidents"], name="Accidents",
               marker_color="#3498db", opacity=0.7), secondary_y=False
    )
    fig_hourly.add_trace(
        go.Scatter(x=hourly["heure"], y=hourly["taux_mortalite"],
                   name="Taux mortalité (%)", line=dict(color="#e74c3c", width=2.5),
                   mode="lines+markers"), secondary_y=True
    )
    fig_hourly.update_layout(
        title="Profil horaire : Volume & Taux de mortalité",
        height=350, legend=dict(x=0.01, y=0.99)
    )
    fig_hourly.update_yaxes(title_text="Nombre d'accidents", secondary_y=False)
    fig_hourly.update_yaxes(title_text="Taux mortalité (%)", secondary_y=True)

    # Évolution par gravité et année
    annual = (
        filtered.assign(annee=pd.to_datetime(filtered["date_accident"]).dt.year)
        .groupby(["annee", "gravite"]).size().reset_index(name="count")
    )
    fig_annual = px.bar(
        annual, x="annee", y="count", color="gravite",
        color_discrete_map=GRAVITE_COLORS, barmode="stack",
        title="Évolution annuelle par gravité",
        labels={"annee": "Année", "count": "Nb accidents", "gravite": "Gravité"}
    )
    fig_annual.update_layout(height=300)

    return html.Div([
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_heatmap))], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_hourly), md=8),
            dbc.Col(dcc.Graph(figure=fig_annual), md=4),
        ])
    ])


def render_geo(filtered: pd.DataFrame) -> html.Div:
    """Onglet Cartographie."""
    # Scatter map
    sample = filtered.sample(min(5000, len(filtered)), random_state=42)

    fig_map = px.scatter_mapbox(
        sample,
        lat="latitude", lon="longitude",
        color="gravite",
        color_discrete_map=GRAVITE_COLORS,
        size_max=8, opacity=0.6,
        zoom=6, center={"lat": 8.0, "lon": 1.1},
        mapbox_style="carto-positron",
        hover_data=["ville", "region", "cause", "type_accident"],
        title="Carte des accidents (échantillon 5000)"
    )
    fig_map.update_layout(height=500, margin=dict(t=40, b=0, l=0, r=0))

    # Table top villes
    top_villes = (
        filtered.groupby("ville")
        .agg(
            accidents=("accident_id", "count"),
            deces=("nombre_deces", "sum"),
            region=("region", "first")
        ).reset_index()
        .sort_values("accidents", ascending=False)
        .head(15)
    )
    top_villes["taux_mort"] = (top_villes["deces"] / top_villes["accidents"] * 100).round(1)

    table = dash_table.DataTable(
        data=top_villes.to_dict("records"),
        columns=[{"name": c, "id": c} for c in top_villes.columns],
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
        style_header={"backgroundColor": THEME_COLOR, "color": "white", "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"filter_query": "{taux_mort} > 10"},
             "backgroundColor": "#fde8e8", "color": "#c0392b"}
        ],
        page_size=10, sort_action="native",
    )

    return html.Div([
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_map))], className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.H5("Top 15 villes les plus accidentogènes", className="fw-bold mb-2"),
                table
            ])
        ])
    ])


def render_causes(filtered: pd.DataFrame) -> html.Div:
    """Onglet Causes & Risques."""
    # Causes par gravité
    causes_grav = (
        filtered.groupby(["cause", "gravite"]).size()
        .reset_index(name="count")
    )
    fig_causes = px.bar(
        causes_grav, x="cause", y="count", color="gravite",
        color_discrete_map=GRAVITE_COLORS, barmode="stack",
        title="Distribution des causes par gravité",
        labels={"cause": "Cause", "count": "Nb accidents", "gravite": "Gravité"}
    )
    fig_causes.update_layout(height=350, xaxis_tickangle=-30)

    # Impact EPI
    epi_data = (
        filtered.groupby(["port_casque", "port_ceinture"])
        .agg(accidents=("accident_id", "count"),
             mortel=("gravite", lambda x: (x == "Mortel").sum()))
        .reset_index()
    )
    epi_data["taux_mortalite"] = epi_data["mortel"] / epi_data["accidents"] * 100
    epi_data["label"] = "Casque:" + epi_data["port_casque"] + " / Ceinture:" + epi_data["port_ceinture"]
    fig_epi = px.bar(
        epi_data.sort_values("taux_mortalite", ascending=False),
        x="label", y="taux_mortalite",
        color="taux_mortalite", color_continuous_scale="YlOrRd",
        title="Taux de mortalité selon le port des EPI",
        labels={"label": "Port EPI", "taux_mortalite": "Taux mortalité (%)"}
    )
    fig_epi.update_layout(height=300, xaxis_tickangle=-20)

    # Vitesse estimée vs limitation
    fig_speed = px.box(
        filtered, x="gravite", y="vitesse_estimee",
        color="gravite", color_discrete_map=GRAVITE_COLORS,
        category_orders={"gravite": GRAVITES},
        title="Distribution de la vitesse estimée par gravité",
        points=False
    )
    fig_speed.update_layout(height=300, showlegend=False)

    # Météo vs gravité
    meteo_grav = (
        filtered.groupby(["meteo", "gravite"]).size()
        .reset_index(name="count")
    )
    fig_meteo = px.bar(
        meteo_grav, x="meteo", y="count", color="gravite",
        color_discrete_map=GRAVITE_COLORS, barmode="group",
        title="Conditions météo et gravité",
        labels={"meteo": "Météo", "count": "Nb accidents"}
    )
    fig_meteo.update_layout(height=300)

    return html.Div([
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_causes))], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_epi), md=6),
            dbc.Col(dcc.Graph(figure=fig_speed), md=6),
        ], className="mb-3"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_meteo))])
    ])


def render_economic(filtered: pd.DataFrame) -> html.Div:
    """Onglet Analyse économique."""
    # Coût par région
    cout_region = (
        filtered.groupby("region")["cout_estime"]
        .agg(["sum", "mean", "count"])
        .reset_index()
    )
    cout_region.columns = ["region", "total", "moyen", "nb"]
    cout_region["total_M"] = cout_region["total"] / 1e6

    fig_cout_region = px.bar(
        cout_region.sort_values("total_M", ascending=True),
        x="total_M", y="region", orientation="h",
        color="total_M", color_continuous_scale="Blues",
        title="Coût total par région (millions FCFA)",
        labels={"total_M": "Coût (MFCFA)", "region": "Région"}
    )
    fig_cout_region.update_layout(height=300)

    # Coût par cause
    cout_cause = (
        filtered.groupby("cause")["cout_estime"]
        .agg(["sum", "mean", "count"])
        .reset_index()
    )
    cout_cause.columns = ["cause", "total", "moyen", "nb"]
    cout_cause["cout_moyen_K"] = cout_cause["moyen"] / 1e3

    fig_cout_cause = px.scatter(
        cout_cause, x="nb", y="cout_moyen_K",
        size="total", color="cause", text="cause",
        title="Cause : Fréquence vs Coût moyen (taille = coût total)",
        labels={"nb": "Nb accidents", "cout_moyen_K": "Coût moyen (kFCFA)"}
    )
    fig_cout_cause.update_traces(textposition="top center")
    fig_cout_cause.update_layout(height=400, showlegend=False)

    # Distribution du coût par gravité
    fig_box_cout = px.violin(
        filtered, x="gravite", y="cout_estime",
        color="gravite", color_discrete_map=GRAVITE_COLORS,
        category_orders={"gravite": GRAVITES},
        title="Distribution du coût estimé par gravité",
        box=True, points=False
    )
    fig_box_cout.update_layout(height=350, showlegend=False)

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_cout_region), md=6),
            dbc.Col(dcc.Graph(figure=fig_box_cout), md=6),
        ], className="mb-3"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_cout_cause))])
    ])


def render_ml_predictor() -> html.Div:
    """Onglet Prédiction ML — formulaire de prédiction de gravité."""
    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.I(className="bi bi-robot me-2"),
                html.Strong("Prédicteur de Gravité d'Accident")
            ]),
            dbc.CardBody([
                dbc.Alert([
                    html.I(className="bi bi-info-circle me-2"),
                    "Renseignez les caractéristiques de l'accident pour prédire sa gravité potentielle."
                ], color="info", className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        html.Label("Heure de l'accident", className="fw-semibold"),
                        dcc.Slider(id="pred-heure", min=0, max=23, step=1, value=14,
                                   marks={i: str(i) for i in range(0, 24, 3)},
                                   tooltip={"placement": "bottom"})
                    ], md=6),
                    dbc.Col([
                        html.Label("Vitesse estimée (km/h)", className="fw-semibold"),
                        dcc.Slider(id="pred-vitesse", min=0, max=200, step=5, value=80,
                                   marks={i: str(i) for i in range(0, 201, 40)},
                                   tooltip={"placement": "bottom"})
                    ], md=6),
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        html.Label("Limitation de vitesse (km/h)", className="fw-semibold"),
                        dcc.Dropdown(id="pred-limitation",
                                     options=[{"label": f"{v} km/h", "value": v}
                                              for v in [30, 50, 70, 90, 110]],
                                     value=70)
                    ], md=4),
                    dbc.Col([
                        html.Label("Type de route", className="fw-semibold"),
                        dcc.Dropdown(id="pred-type-route",
                                     options=[{"label": r, "value": r}
                                              for r in ["Urbaine", "Interurbaine", "Rurale"]],
                                     value="Urbaine")
                    ], md=4),
                    dbc.Col([
                        html.Label("Météo", className="fw-semibold"),
                        dcc.Dropdown(id="pred-meteo",
                                     options=[{"label": m, "value": m}
                                              for m in ["Soleil", "Nuageux", "Pluie", "Brouillard"]],
                                     value="Soleil")
                    ], md=4),
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        html.Label("Cause principale", className="fw-semibold"),
                        dcc.Dropdown(id="pred-cause",
                                     options=[{"label": c, "value": c} for c in [
                                         "Alcool", "Téléphone", "Fatigue", "Excès de vitesse",
                                         "Dépassement dangereux", "Priorité non respectée", "Freinage brusque"
                                     ]], value="Excès de vitesse")
                    ], md=4),
                    dbc.Col([
                        html.Label("Port du casque", className="fw-semibold"),
                        dcc.RadioItems(id="pred-casque",
                                       options=[{"label": " Oui", "value": 1}, {"label": " Non", "value": 0}],
                                       value=1, inline=True, inputStyle={"marginRight": "5px"})
                    ], md=4),
                    dbc.Col([
                        html.Label("Luminosité", className="fw-semibold"),
                        dcc.Dropdown(id="pred-luminosite",
                                     options=[{"label": l, "value": l}
                                              for l in ["Jour", "Nuit", "Crépuscule"]],
                                     value="Jour")
                    ], md=4),
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="bi bi-cpu me-2"), "Prédire la gravité"],
                            id="btn-predict", color="primary", size="lg", className="w-100"
                        )
                    ], md=4, className="mx-auto")
                ], className="mb-3"),

                html.Div(id="prediction-result")
            ])
        ])
    ])


@app.callback(
    Output("prediction-result", "children"),
    Input("btn-predict", "n_clicks"),
    [State("pred-heure", "value"),
     State("pred-vitesse", "value"),
     State("pred-limitation", "value"),
     State("pred-type-route", "value"),
     State("pred-meteo", "value"),
     State("pred-cause", "value"),
     State("pred-casque", "value"),
     State("pred-luminosite", "value")],
    prevent_initial_call=True
)
def predict_severity(n_clicks, heure, vitesse, limitation, type_route, meteo, cause, casque, luminosite):
    """Prédiction de la gravité basée sur les entrées utilisateur (règles expertes simulées)."""
    # Score de risque basé sur règles expertes (en attendant le modèle ML)
    score = 0
    details = []

    # Excès de vitesse
    exces = max(0, vitesse - limitation)
    if exces > 40:
        score += 3
        details.append(f"⚠️ Excès de vitesse critique : +{exces:.0f} km/h")
    elif exces > 20:
        score += 2
        details.append(f"⚠️ Excès de vitesse modéré : +{exces:.0f} km/h")
    elif exces > 0:
        score += 1
        details.append(f"Excès de vitesse léger : +{exces:.0f} km/h")

    # Nuit
    if luminosite == "Nuit" or heure < 5 or heure >= 22:
        score += 2
        details.append("🌙 Conduite de nuit")

    # Météo
    if meteo in ["Pluie", "Brouillard"]:
        score += 2
        details.append(f"🌧️ Conditions météo défavorables : {meteo}")

    # Cause
    if cause in ["Alcool", "Fatigue"]:
        score += 3
        details.append(f"🍺 Cause à risque élevé : {cause}")
    elif cause == "Téléphone":
        score += 2
        details.append("📱 Distraction au téléphone")

    # EPI
    if casque == 0:
        score += 2
        details.append("🏍️ Pas de casque")

    # Route rurale
    if type_route == "Rurale":
        score += 1
        details.append("🛣️ Route rurale (risque accru)")

    # Classification
    if score <= 2:
        gravite, color, icon = "Léger", "success", "check-circle"
        proba = [0.70, 0.22, 0.08]
    elif score <= 5:
        gravite, color, icon = "Grave", "warning", "exclamation-triangle"
        proba = [0.25, 0.55, 0.20]
    else:
        gravite, color, icon = "Mortel", "danger", "x-circle"
        proba = [0.08, 0.25, 0.67]

    # Jauge de probabilités
    fig_proba = go.Figure(go.Bar(
        x=["Léger", "Grave", "Mortel"], y=proba,
        marker_color=["#27ae60", "#f39c12", "#e74c3c"],
        text=[f"{p:.0%}" for p in proba], textposition="outside"
    ))
    fig_proba.update_layout(
        title="Probabilités estimées",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=250, margin=dict(t=40, b=10)
    )

    return dbc.Card([
        dbc.CardHeader(html.H5([
            html.I(className=f"bi bi-{icon} me-2"),
            f"Gravité prédite : {gravite}"
        ], className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Alert(f"Score de risque : {score}/12", color=color, className="fw-bold text-center"),
                    html.H6("Facteurs de risque identifiés :", className="fw-semibold"),
                    html.Ul([html.Li(d) for d in details] if details else [html.Li("Aucun facteur de risque majeur")])
                ], md=6),
                dbc.Col([dcc.Graph(figure=fig_proba)], md=6)
            ])
        ])
    ], color=color, outline=True, className="mt-3")


# ─────────────────────────────────────────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚦 Togo Road Safety Dashboard")
    print("   → http://localhost:8050")
    app.run(debug=True, host="0.0.0.0", port=8050)
