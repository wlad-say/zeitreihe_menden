"""
layout/tab_gold.py
Tab-Layout für die Gold-Analyse.
"""

from dash import html
from .components import card, section_title, info_block, plot_image, results_table


def render(data: dict) -> html.Div:
    p = data["plots"]
    m = data["metrics"]

    return html.Div(style={"padding": "24px"}, children=[

        card(
            section_title("1 · Zeitreihe & EDA"),
            info_block(
                "Goldpreise zeigen starken Aufwärtstrend — Eurokrise (2011) und "
                "Corona-Beginn (2020) als deutliche Ausschläge sichtbar.  \n"
                "Verteilung **rechtsschief** mit Extremwerten > 2000 USD.  \n"
                "Saisondekomposition: **Trend dominiert**, saisonale Komponente gering."
            ),
            plot_image(p["ts"]),
            plot_image(p["eda"]),
        ),

        card(
            section_title("2 · Stationarität & Transformation"),
            info_block(
                "**ADF** (alle drei Varianten): p >> 0.05 — Einheitswurzel vorhanden.  \n"
                "**KPSS**: p = 0.01 — Stationarität abgelehnt.  \n"
                "→ Erste Differenz macht Reihe stationär (ADF p ≈ 0.0).  \n"
                "**Integrationsordnung: d = 1.** Log-Returns für ergänzende Analyse berechnet."
            ),
            plot_image(p["lr"]),
        ),

        card(
            section_title("3 · Modellidentifikation (ACF/PACF) & Selektion"),
            info_block(
                "Differenzierte Reihe zeigt kleine frühe Ausschläge → einfache ARIMA-Ordnungen geeignet.  \n"
                "**Grid-Search** (p, q ∈ {0…3}, d = 1): ARIMA(3,1,3) bestes AIC/BIC.  \n"
                "Finales Modell: **ARIMA(2,1,2)** — numerisch stabiler, ähnliche Güte."
            ),
            plot_image(p["acf"]),
        ),

        card(
            section_title("4 · Residualanalyse & 10-Tage-Forecast"),
            results_table([
                ("Modell",        "ARIMA(2,1,2)"),
                ("Ljung-Box (Lag 10)", f"p = {m['lb_p']:.3f} — keine Autokorrelation ✓"),
                ("Jarque-Bera",   f"p = {m['jb_p']:.4f} — Fat Tails (erwartet) ✗"),
            ]),
            info_block(
                "Residuen ohne systematisches Muster — Modell erfasst zeitliche Abhängigkeiten gut.  \n"
                "Nicht-Normalität durch Fat Tails bei Finanzdaten strukturell erwartet.  \n"
                "10-Tage-Forecast zeigt typische Konvergenz zum Mittelwert mit wachsendem "
                "Konfidenzintervall."
            ),
            plot_image(p["resid"]),
            plot_image(p["fc"]),
        ),

    ])
