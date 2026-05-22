"""
kpi_cards.py
============
Composants KPI réutilisables pour le Dashboard Dash.
"""

import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html, dcc
from typing import Optional


GRAVITE_COLORS = {"Léger": "#27ae60", "Grave": "#f39c12", "Mortel": "#e74c3c"}
PALETTE        = {"primary": "#2c3e50", "secondary": "#3498db",
                  "success": "#27ae60", "warning": "#f39c12",
                  "danger": "#e74c3c", "info": "#17a2b8"}

CARD_STYLE = {
    "borderRadius": "10px",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.08)",
    "border": "none",
    "transition": "all 0.25s ease",
}


# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────

def kpi_card(title: str, value: str, icon: str, color: str,
             subtitle: str = "", trend: Optional[str] = None) -> dbc.Card:
    """
    Carte KPI avec valeur, icône et sous-titre optionnel.

    Parameters
    ----------
    title    : Libellé de l'indicateur
    value    : Valeur à afficher (string formaté)
    icon     : Nom d'icône Bootstrap Icons (ex: 'car-front', 'heart-pulse')
    color    : Couleur hex ou CSS
    subtitle : Texte secondaire (ex: 'Taux: 8.2%')
    trend    : Indicateur de tendance '▲ +5%' ou '▼ -3%'
    """
    trend_el = html.Span()
    if trend:
        trend_color = PALETTE["success"] if "▲" in trend or "+" in trend else PALETTE["danger"]
        trend_el = html.Small(trend, style={"color": trend_color, "fontSize": "0.75rem", "fontWeight": "600"})

    return dbc.Card([
        dbc.CardBody([
            html.Div([
                # Colonne texte
                html.Div([
                    html.H3(value, className="mb-0 fw-bold",
                            style={"color": color, "fontSize": "1.8rem", "lineHeight": "1"}),
                    html.P(title, className="text-muted mb-0 mt-1",
                           style={"fontSize": "0.78rem", "textTransform": "uppercase", "letterSpacing": "0.5px"}),
                    html.Div([
                        html.Small(subtitle, className="text-muted") if subtitle else html.Span(),
                        trend_el,
                    ], style={"display": "flex", "gap": "8px", "alignItems": "center"})
                ], style={"flex": "1"}),
                # Icône
                html.Div([
                    html.I(className=f"bi bi-{icon}",
                           style={"fontSize": "2.2rem", "color": color, "opacity": "0.25"})
                ], style={"alignSelf": "center"})
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"})
        ], className="py-3 px-3")
    ], style=CARD_STYLE, className="h-100")


def kpi_row_from_df(df: pd.DataFrame) -> list:
    """
    Génère la rangée complète de KPIs depuis un DataFrame filtré.
    Retourne une liste de dbc.Col contenant les cartes.
    """
    total    = len(df)
    deces    = int(df["nombre_deces"].sum())
    victimes = int(df["nombre_victimes"].sum()) if "nombre_victimes" in df else 0
    tx_mort  = round(deces / total * 100, 1) if total > 0 else 0
    cout_G   = df["cout_estime"].sum() / 1e9 if "cout_estime" in df else 0
    nb_reg   = df["region"].nunique()
    moy_int  = df["intervention_secours"].mean() if "intervention_secours" in df else 0

    cards = [
        kpi_card("Total accidents",    f"{total:,}",         "car-front",    PALETTE["secondary"]),
        kpi_card("Décès",              f"{deces:,}",          "heart-pulse",  PALETTE["danger"],
                 subtitle=f"Taux: {tx_mort:.1f}%"),
        kpi_card("Victimes",           f"{victimes:,}",       "people-fill",  PALETTE["warning"]),
        kpi_card("Coût estimé",        f"{cout_G:.2f} G",     "cash-stack",   PALETTE["success"],
                 subtitle="Milliards FCFA"),
        kpi_card("Régions",            str(nb_reg),           "geo-alt",      "#9b59b6"),
        kpi_card("Délai intervention", f"{moy_int:.0f} min",  "clock",        PALETTE["info"],
                 subtitle="Moyenne"),
    ]

    return [dbc.Col(card, md=2, className="mb-2") for card in cards]


# ─────────────────────────────────────────────────────────────────────────────
# INDICATEURS GAUGE (Plotly)
# ─────────────────────────────────────────────────────────────────────────────

def gauge_taux_mortalite(taux: float) -> go.Figure:
    """Indicateur gauge du taux de mortalité."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=taux,
        number={"suffix": "%", "font": {"size": 36, "color": PALETTE["danger"]}},
        title={"text": "Taux de mortalité", "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 30], "tickwidth": 1, "tickcolor": "#aaa"},
            "bar": {"color": PALETTE["danger"], "thickness": 0.35},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#eee",
            "steps": [
                {"range": [0, 10],  "color": "#eafaf1"},
                {"range": [10, 20], "color": "#fef9e7"},
                {"range": [20, 30], "color": "#fdedec"},
            ],
            "threshold": {
                "line": {"color": "#c0392b", "width": 4},
                "thickness": 0.75,
                "value": taux
            }
        }
    ))
    fig.update_layout(height=220, margin=dict(t=30, b=10, l=20, r=20),
                      paper_bgcolor="white", font=dict(family="Segoe UI"))
    return fig


def indicator_number(value: float, title: str, prefix: str = "",
                     suffix: str = "", color: str = "#3498db") -> go.Figure:
    """Indicateur numérique simple (Plotly Indicator)."""
    fig = go.Figure(go.Indicator(
        mode="number",
        value=value,
        number={"prefix": prefix, "suffix": suffix,
                "font": {"size": 42, "color": color}},
        title={"text": title, "font": {"size": 13, "color": "#555"}}
    ))
    fig.update_layout(height=150, margin=dict(t=20, b=10, l=10, r=10),
                      paper_bgcolor="white", plot_bgcolor="white")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ALERTS & BADGES
# ─────────────────────────────────────────────────────────────────────────────

def alert_no_data() -> dbc.Alert:
    """Alerte affichée quand aucune donnée ne correspond aux filtres."""
    return dbc.Alert([
        html.I(className="bi bi-exclamation-triangle-fill me-2"),
        "Aucun accident ne correspond aux filtres sélectionnés. Essayez d'élargir la sélection."
    ], color="warning", className="mt-3")


def badge_gravite(gravite: str) -> html.Span:
    """Badge coloré selon la gravité."""
    color_map = {"Léger": "success", "Grave": "warning", "Mortel": "danger"}
    color = color_map.get(gravite, "secondary")
    return html.Span(gravite, className=f"badge bg-{color}",
                     style={"fontSize": "0.75rem"})


def info_box(title: str, content: str, color: str = "info") -> dbc.Card:
    """Boîte d'information contextuelle."""
    border_color = {
        "info": "#3498db", "warning": "#f39c12",
        "danger": "#e74c3c", "success": "#27ae60"
    }.get(color, "#3498db")

    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="fw-bold mb-1", style={"color": border_color}),
            html.P(content, className="mb-0", style={"fontSize": "0.875rem"})
        ], className="py-2 px-3")
    ], style={
        **CARD_STYLE,
        "borderLeft": f"4px solid {border_color}",
        "borderRadius": "0 10px 10px 0"
    })


# ─────────────────────────────────────────────────────────────────────────────
# RÉSUMÉ STATISTIQUE RAPIDE
# ─────────────────────────────────────────────────────────────────────────────

def quick_stats_table(df: pd.DataFrame) -> dbc.Table:
    """Tableau de statistiques rapides."""
    total  = len(df)
    mortel = (df["gravite"] == "Mortel").sum()
    grave  = (df["gravite"] == "Grave").sum()
    leger  = (df["gravite"] == "Léger").sum()

    rows = [
        ("Accidents totaux",         f"{total:,}",  ""),
        ("Accidents mortels",        f"{mortel:,}",  f"{mortel/total*100:.1f}%"),
        ("Accidents graves",         f"{grave:,}",   f"{grave/total*100:.1f}%"),
        ("Accidents légers",         f"{leger:,}",   f"{leger/total*100:.1f}%"),
        ("Vitesse moyenne",          f"{df['vitesse_estimee'].mean():.0f} km/h", ""),
        ("Coût moyen",               f"{df['cout_estime'].mean()/1e3:.0f} kFCFA", ""),
        ("Intervention moyenne",     f"{df['intervention_secours'].mean():.0f} min", ""),
    ]

    table_body = [html.Tbody([
        html.Tr([
            html.Td(label, style={"fontSize": "0.82rem", "color": "#555"}),
            html.Td(html.Strong(value), style={"fontSize": "0.85rem"}),
            html.Td(pct, style={"fontSize": "0.8rem", "color": "#888"}),
        ]) for label, value, pct in rows
    ])]

    return dbc.Table(table_body, bordered=False, hover=True, responsive=True,
                     size="sm", className="mb-0")
