"""
layout/tab_weather.py
Tab-Layout für die Wetterdaten-Analyse.
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
                "Tägliche Temperaturdaten 2009–2017, aggregiert aus 10-Minuten-Messungen.  \n"
                "Klare **Jahressaisonalität** — Sommermaxima ~35 °C, Winterminima bis −20 °C.  \n"
                "8 Ausreißer (IQR-Regel) — extreme Kälteereignisse, keine Messfehler "
                "→ nicht entfernt."
            ),
            plot_image(p["ts"]),
            plot_image(p["eda"]),
        ),

        card(
            section_title("2 · Stationarität & ACF/PACF"),
            info_block(
                "**ADF p = 0.006, KPSS p = 0.1** → Originalreihe stationär → **d = 0**.  \n"
                "ACF (400 Lags): sinusförmiger Verlauf, Spike bei Lag 365 bestätigt "
                "Jahressaisonalität.  \n"
                "PACF: Spike bei Lag 1 (~0.93), klingt nach Lag 3 ab → klares **AR(3)-Muster**.  \n"
                "Saisonalität wird über **Fourier-Terms** (K = 3, Periode = 365.25) "
                "als exogene Variable modelliert."
            ),
            plot_image(p["acf"]),
        ),

        card(
            section_title("3 · Residualanalyse – ARIMA(3,0,0) + Fourier"),
            results_table([
                ("Ljung-Box (Lag 30)",  f"p = {m['lb_p']:.3f} ✓  — Residuen unkorreliert"),
                ("Jarque-Bera",         "p < 0.05  — nicht normalverteilt (Kälteextreme)"),
                ("ACF Residuen",         "alle Lags im Konfidenzband ✓"),
            ]),
            info_block(
                "Das Modell hat die zeitliche Abhängigkeitsstruktur vollständig erfasst.  \n"
                "Nicht-Normalität der Residuen ist durch die 8 extremen Kälteereignisse "
                "inhaltlich begründbar — kein Modellversagen."
            ),
            plot_image(p["resid"]),
        ),

        card(
            section_title("4 · Testset-Evaluation & 10-Tage-Forecast"),
            results_table([
                ("Modell",  "ARIMA(3,0,0) + Fourier-Terms (K = 3)"),
                ("RMSE",    f"{m['rmse']:.2f} °C"),
                ("MAE",     f"{m['mae']:.2f} °C"),
                ("Prophet (Vergleich)", "RMSE Δ = 0.28 °C — vernachlässigbar"),
            ]),
            info_block(
                "RMSE ~4 °C bei 10-Tage-Horizont meteorologisch plausibel — kurzfristige "
                "Wettervolatilität ist strukturell nicht prognostizierbar.  \n"
                "**Entscheidung für ARIMA(3,0,0):** Lehrplankonformität (Box-Jenkins), "
                "vollständige statistische Validierbarkeit, Sparsamkeitsprinzip."
            ),
            plot_image(p["test"]),
            plot_image(p["fc"]),
        ),

    ])
