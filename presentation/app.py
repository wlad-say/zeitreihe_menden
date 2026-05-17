"""
presentation/app.py
Einstiegspunkt der Präsentations-App.

Starten:  python presentation/app.py
Browser:  http://127.0.0.1:8050
"""

import sys, os

_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

import dash
from dash import dcc, html, Input, Output

from data_loader import load_apple, load_weather, load_gold
from plots       import apple     as plot_apple
from plots       import weather   as plot_weather
from plots       import gold      as plot_gold
from plots       import var_model as plot_var
from layout      import tab_apple, tab_weather, tab_gold, tab_var
from layout.components import (
    STYLE_PAGE, STYLE_TABS, STYLE_TAB, STYLE_TAB_SELECTED,
    STYLE_CONTENT, COLORS,
)


# ── Daten & Modelle ──────────────────────────────────────────────────────────
print("Lade Datensätze …")
df_apple   = load_apple()
df_weather = load_weather()
df_gold    = load_gold()
print("  Daten geladen. Berechne Modelle und Plots …")

data_apple   = plot_apple.build(df_apple);    print("  Apple fertig")
data_weather = plot_weather.build(df_weather); print("  Weather fertig")
data_gold    = plot_gold.build(df_gold);      print("  Gold fertig")
data_var     = plot_var.build(df_apple, df_gold, df_weather)
print("  VAR fertig — starte Dash-Server\n")


# ── App ──────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Zeitreihenanalyse – Team 03")

# Browser-Default-Margin/Padding komplett entfernen
app.index_string = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      * { box-sizing: border-box; }
      html, body { margin: 0; padding: 0; background: #eef0f3; }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
  </body>
</html>"""

app.layout = html.Div(style=STYLE_PAGE, children=[

    # ── Header ───────────────────────────────────────────────────────────────
    html.Div(
        style={
            "background": "linear-gradient(135deg, #1a2535 0%, #2e5f8a 100%)",
            "padding":    "22px 36px 18px 36px",
        },
        children=[
            html.Div(
                style={"maxWidth": "1200px", "margin": "0 auto"},
                children=[
                    html.Div(
                        "THWS · Business Analytics · Zeitreihenanalyse",
                        style={
                            "fontSize":      "0.72rem",
                            "color":         "#a8c4e0",
                            "letterSpacing": "0.1em",
                            "textTransform": "uppercase",
                            "marginBottom":  "4px",
                        },
                    ),
                    html.H1(
                        "Zeitreihenanalyse",
                        style={
                            "color":         "#ffffff",
                            "margin":        "0 0 4px 0",
                            "fontWeight":    "600",
                            "fontSize":      "1.55rem",
                            "letterSpacing": "0.01em",
                            "fontFamily":    "'Segoe UI', Arial, sans-serif",
                        },
                    ),
                    html.P(
                        "Apple Stock  ·  Wetterdaten  ·  Goldpreis  ·  Multivariate VAR-Analyse",
                        style={
                            "color":      "#b8d4ea",
                            "margin":     "0",
                            "fontSize":   "0.85rem",
                            "fontFamily": "'Segoe UI', Arial, sans-serif",
                        },
                    ),
                ],
            ),
        ],
    ),

    # ── Tabs ─────────────────────────────────────────────────────────────────
    html.Div(
        style={
            "background":   "#eef0f3",
            "borderBottom": "1px solid #cdd3da",
        },
        children=[
            html.Div(
                style={"maxWidth": "1200px", "margin": "0 auto"},
                children=[
                    dcc.Tabs(
                        id="tabs",
                        value="apple",
                        style=STYLE_TABS,
                        children=[
                            dcc.Tab(label="Apple Stock", value="apple",
                                    style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                            dcc.Tab(label="Wetterdaten", value="weather",
                                    style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                            dcc.Tab(label="Goldpreis",   value="gold",
                                    style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                            dcc.Tab(label="VAR-Analyse", value="var",
                                    style=STYLE_TAB, selected_style=STYLE_TAB_SELECTED),
                        ],
                    ),
                ],
            ),
        ],
    ),

    # ── Tab-Inhalt ────────────────────────────────────────────────────────────
    html.Div(
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "24px 28px 60px 28px"},
        children=[html.Div(id="tab-content")],
    ),
])


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab: str):
    if tab == "apple":   return tab_apple.render(data_apple)
    if tab == "weather": return tab_weather.render(data_weather)
    if tab == "gold":    return tab_gold.render(data_gold)
    if tab == "var":     return tab_var.render(data_var)


if __name__ == "__main__":
    app.run(debug=False, port=8050)