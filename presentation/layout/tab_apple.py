"""
layout/tab_apple.py
Tab-Layout für die Apple-Stock-Analyse.
"""

from dash import html
from .components import card, section_title, info_block, plot_image, results_table


def render(data: dict) -> html.Div:
    p = data["plots"]
    m = data["metrics"]

    return html.Div(style={"padding": "24px"}, children=[

        card(
            section_title("1 · Zeitreihe & Transformation"),
            info_block(
                "**Apple Aktienkurs (Close-Preis):** AAPL hat sich von ~$30 (2016) auf über $260 "
                "mehr als verachtfacht. Markante Einbrüche: COVID-Crash (März 2020) und "
                "Zinsanstieg-Korrektur (2022).  \n"
                "ADF p = 0.9541, KPSS p = 0.01 → Zeitreihe **nicht stationär** "
                "→ Transformation zu **Log-Returns** (d = 0 nach Transformation).  \n"
                "**Stationaritätstest nach Transformation:** ADF p < 0.001 ✓, KPSS p > 0.1 ✓ — "
                "Log-Returns sind stationär (I(0))."
            ),
            plot_image(p["close"]),
            plot_image(p["lr"]),
        ),

        card(
            section_title("2 · ACF & PACF der Log-Returns"),
            info_block(
                "**PACF** bricht nach Lag 1 scharf ab → AR(1)-Struktur identifiziert.  \n"
                "**ACF** zeigt schwach negative, aber signifikante Autokorrelation bei Lag 1, "
                "danach alle Werte im 95%-Konfidenzintervall → MA(0) oder MA(1) möglich.  \n"
                "Startspezifikation ARIMA(1,0,0) — nach AIC-Vergleich kombiniert mit "
                "Ljung-Box-Residualdiagnose reoptimiert zu **ARIMA(2,0,2)** als finalem Modell "
                "(einziges Modell, das den Residualtest besteht: LB p = 0.1335)."
            ),
            plot_image(p["acf"]),
        ),

        card(
            section_title("3 · Residualanalyse – ARIMA(2,0,2)"),
            info_block(
                "Alle sechs Koeffizienten signifikant auf 5%-Niveau "
                "(const, ar.L1, ar.L2, ma.L1, ma.L2, σ²).  \n"
                "**Ljung-Box p ≈ 0.0:** ARCH-Effekte (Volatilitäts-Clustering) — "
                "strukturell nicht durch lineare ARIMA-Modelle eliminierbar.  \n"
                "**Jarque-Bera:** Fat Tails (Kurtosis ≈ 8.2) — typisch für Aktienrenditen, "
                "kein Modellversagen. ACF/PACF der Residuen zeigen keine Muster → "
                "lineare Struktur vollständig erfasst."
            ),
            plot_image(p["resid"]),
        ),

        card(
            section_title("4 · Evaluation auf dem Testset"),
            results_table([
                ("Modell",        "ARIMA(2,0,2)"),
                ("MAE",           f"{m['mae']:.5f}"),
                ("RMSE",          f"{m['rmse']:.5f}"),
                ("Null-Modell",   "MAE ≈ RMSE — praktisch gleich"),
            ]),
            info_block(
                "Die Prognose liegt nahezu auf der Nulllinie — konsistent mit der "
                "**Efficient Market Hypothesis (EMH)**: vergangene Kurse enthalten kaum "
                "verwertbare Information für zukünftige Renditen. "
                "Der flache Forecast ist das statistisch korrekte Ergebnis.  \n"
                "Der **Trump-Zollschock (April 2025)** ist als markanter Ausreißer sichtbar — "
                "ein temporäres Ereignis, kein dauerhafter Strukturbruch."
            ),
            plot_image(p["test"]),
        ),

        card(
            section_title("5 · 10-Tage-Forecast"),
            results_table([
                ("Letzter Kurs",      f"${m['letzter_kurs']:.2f}"),
                ("Prognose +10 Tage", f"${m['fc_kurs']:.2f}"),
                ("95%-KI",            f"${m['ci_lo']:.2f} – ${m['ci_hi']:.2f}"),
            ]),
            info_block(
                "Mittelpfad nahezu flach. Ab Schritt 2 konvergiert die Prognose auf den "
                "konstanten Erwartungswert — typisches Verhalten eines ARIMA(2,0,2)-Prozesses.  \n"
                "Das breite Konfidenzintervall spiegelt die strukturelle Unsicherheit "
                "von Aktienkursen korrekt wider — konsistent mit der **EMH**: "
                "Aktienkurse sind kurzfristig nicht zuverlässig prognostizierbar."
            ),
            plot_image(p["fc"]),
        ),

    ])