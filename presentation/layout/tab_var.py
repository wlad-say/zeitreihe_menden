"""
layout/tab_var.py
Tab-Layout für die multivariate VAR-Analyse.
"""

from dash import html
from .components import card, section_title, info_block, plot_image, results_table


def render(data: dict) -> html.Div:
    p = data["plots"]
    m = data["metrics"]

    return html.Div(style={"padding": "24px"}, children=[

        card(
            section_title("1 · Datenbasis & Transformation"),
            results_table([
                ("Variablen",    "Apple Close Log-Return · Gold Log-Return · Δ Temperatur"),
                ("Zeitraum",     "ab 2010 (Inner-Join der drei Datensätze)"),
                ("Beobachtungen", str(m["n_obs"])),
                ("VAR-Lags",     str(m["best_lag"]) + "  (Selektion nach AIC)"),
            ]),
            info_block(
                "Stationarisierung: Apple & Gold → Log-Returns, Temperatur → erste Differenz.  \n"
                "Alle drei transformierten Reihen bestehen ADF- und KPSS-Tests.  \n"
                "VAR-Lag-Selektion nach AIC über maxlags = 10."
            ),
            plot_image(p["ts"]),
        ),

        card(
            section_title("2 · Korrelationsanalyse"),
            info_block(
                "Korrelationen zwischen den transformierten Reihen liegen nahe Null — "
                "wie bei effizienten Märkten zu erwarten.  \n"
                "Apple und Gold zeigen keine linearen Abhängigkeiten zur Temperatur.  \n"
                "Das VAR-Modell kann dennoch dynamische Wechselwirkungen über Lags erfassen."
            ),
            plot_image(p["corr"], style={"maxWidth": "480px", "display": "block",
                                         "margin": "0 auto 18px auto"}),
        ),

        card(
            section_title("3 · Impuls-Antwort-Funktionen (IRF)"),
            info_block(
                "IRF zeigen die Reaktion jeder Variable auf einen orthogonalisierten "
                "Standardschock in einer anderen.  \n"
                "Reaktionen klingen schnell ab — kein persistenter Übertragungseffekt "
                "zwischen den drei Märkten nachweisbar."
            ),
            plot_image(p["irf"]),
        ),

        card(
            section_title("4 · Forecast Error Variance Decomposition (FEVD)"),
            info_block(
                "FEVD quantifiziert, welcher Anteil der Prognoseunsicherheit einer Variable "
                "durch eigene vs. fremde Schocks erklärt wird.  \n"
                "Erwartetes Ergebnis bei unabhängigen Märkten: jede Variable wird primär "
                "durch **eigene Schocks** dominiert."
            ),
            plot_image(p["fevd"]),
        ),

        card(
            section_title("5 · Residuen & 10-Tage-Forecast"),
            info_block(
                "VAR-Residuen schwanken ohne systematisches Muster um null.  \n"
                "10-Tage-Forecast zeigt charakteristische Konvergenz zum Mittelwert — "
                "konsistent mit den Einzelmodellen (ARIMAX, ARIMA, SARIMAX).  \n"
                "Die geringe Korrelation der Variablen begrenzt den Informationsgewinn "
                "des multivariaten Ansatzes gegenüber den Einzelmodellen."
            ),
            plot_image(p["resid"]),
            plot_image(p["fc"]),
        ),

    ])
