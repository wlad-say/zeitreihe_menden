"""
layout/tab_weather.py
Tab-Layout für die Wetterdaten-Analyse.

Neu (v2):
  - Karte 3a: AR-Koeffiziententabelle (Parameter + p-Werte)
  - Karte 5:  Walk-Forward Validation (6 Folds, Boxplots + Mittelwerte)
"""

from dash import html
from .components import card, section_title, info_block, plot_image, results_table


def render(data: dict) -> html.Div:
    p = data["plots"]
    m = data["metrics"]

    # ── Koeffizienten-Tabelle aufbauen ────────────────────────────────────────
    # ar_rows: Liste von (Label, Koeff, p-Wert, Signifikanz-Symbol)
    coef_rows = [
        (label, f"Koeffizient: {coef}  |  p-Wert: {pval}  {sig}")
        for label, coef, pval, sig in m.get("ar_rows", [])
    ]

    # ── Walk-Forward Ergebnisse ───────────────────────────────────────────────
    wf = m.get("wf_means", {})
    wf_rows = [
        ("Modell", "Ø RMSE (6 Folds)  |  Ø MAE (6 Folds)"),
        ("ARIMA(3,0,0) + Fourier",
         f"{wf.get('arima_rmse', '–'):.3f} °C  |  {wf.get('arima_mae', '–'):.3f} °C"),
    ]
    if m.get("prophet_available"):
        wf_rows.append((
            "Prophet (Vergleich)",
            f"{wf.get('prophet_rmse', '–'):.3f} °C  |  {wf.get('prophet_mae', '–'):.3f} °C",
        ))
    wf_rows.append((
        "Seasonal Naive (Baseline)",
        f"{wf.get('naive_rmse', '–'):.3f} °C  |  {wf.get('naive_mae', '–'):.3f} °C",
    ))

    return html.Div(style={"padding": "24px"}, children=[

        # ── Karte 1: Zeitreihe & EDA ──────────────────────────────────────────
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

        # ── Karte 2: Stationarität & ACF/PACF ────────────────────────────────
        card(
            section_title("2 · Stationarität & ACF/PACF"),
            info_block(
                "**ADF p = 0.006, KPSS p = 0.1** → Originalreihe stationär → **d = 0**.  \n"
                "ACF (400 Lags): sinusförmiger Verlauf, Spike bei Lag 365 bestätigt "
                "Jahressaisonalität.  \n"
                "PACF: Spike bei Lag 1 (~0.93), negativer Wert bei Lag 2, kleiner positiver "
                "Wert bei Lag 3 — alle weiteren Lags innerhalb des Konfidenzbands → klares **AR(3)-Muster**.  \n"
                "Saisonalität wird über **Fourier-Terms** (K = 3, Periode = 365.25) "
                "als exogene Variable modelliert."
            ),
            plot_image(p["acf"]),
        ),

        # ── Karte 3: Residualanalyse ──────────────────────────────────────────
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

        # ── Karte 3a: NEU — AR-Koeffizienten ─────────────────────────────────
        card(
            section_title("3a · AR-Koeffizienten & Signifikanz"),
            results_table(coef_rows) if coef_rows else html.Div(),
            info_block(
                "**AR(L1) ≈ +0.99:** Sehr starke positive Autokorrelation zum Vortag — "
                "die Temperatur von gestern dominiert die heutige Prognose.  \n"
                "**AR(L2) ≈ −0.27:** Leichte Korrektur, verhindert explosive Dynamik.  \n"
                "**AR(L3) ≈ +0.08:** Kleiner, aber statistisch signifikanter Drittlags-Effekt.  \n"
                "Alle drei AR-Parameter sind hochsignifikant (p = 0.000) — konsistent mit "
                "dem PACF-Befund (Spikes bei Lag 1–3)."
            ),
        ),

        # ── Karte 4: Testset-Evaluation & Forecast ────────────────────────────
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

        # ── Karte 5: NEU — Walk-Forward Validation ────────────────────────────
        card(
            section_title("5 · Walk-Forward Validation (6 Folds)"),
            info_block(
                "Ein einzelner Hold-out-Split ist für Zeitreihen nur eingeschränkt "
                "aussagekräftig — ein zufällig stabiler Testzeitraum kann ein Modell "
                "besser erscheinen lassen als es wirklich ist.  \n"
                "**Expanding-Window-Schema:** 6 Folds in 60-Tage-Schritten decken "
                "unterschiedliche Jahreszeiten und meteorologische Regime ab.  \n"
                "**Seasonal Naive** (Vorhersage = Temperatur vor exakt 365 Tagen) dient "
                "als Baseline — ohne Baseline ist kein echter Mehrwert messbar."
            ),
            results_table(wf_rows),
            plot_image(p["wf"]),
        ),

    ])