"""
layout/tab_gold.py
Tab-Layout für die Gold-Analyse.
Zeigt alle Grafiken und Ergebnisse aus plots/gold.py in richtiger Reihenfolge.
"""

from dash import html
from .components import card, section_title, info_block, plot_image, results_table


def render(data: dict) -> html.Div:
    p = data["plots"]
    m = data["metrics"]

    return html.Div(style={"padding": "24px"}, children=[

        # ============================================================
        # 1 · Zeitreihe
        # ============================================================
        card(
            section_title("1 · Goldpreis-Zeitreihe"),
            info_block(
                "Die Zeitreihe zeigt die Entwicklung des Goldpreises über den gesamten "
                "Beobachtungszeitraum. Besonders auffällig sind starke Ausschläge rund um "
                "die Eurokrise 2011 sowie den Corona-Beginn 2020. Insgesamt ist ein klarer "
                "langfristiger Aufwärtstrend sichtbar."
            ),
            plot_image(p["ts"]),
        ),

        # ============================================================
        # 2 · Explorative Datenanalyse
        # ============================================================
        card(
            section_title("2 · Explorative Datenanalyse"),
            info_block(
                "Das Histogramm zeigt die Verteilung der Goldpreise. Die Verteilung ist "
                "rechtsschief, da einige sehr hohe Goldpreise auftreten. Der Monatsboxplot "
                "zeigt Unterschiede in Median und Streuung zwischen den Monaten. Ein stark "
                "ausgeprägtes saisonales Muster ist jedoch nicht eindeutig erkennbar."
            ),
            plot_image(p["eda"]),
        ),

        # ============================================================
        # 3 · Log-Returns
        # ============================================================
        card(
            section_title("3 · Transformation: Log-Returns"),
            info_block(
                "Da Finanzzeitreihen häufig Trends und starke Schwankungen enthalten, "
                "werden zusätzlich Log-Returns berechnet. Die ursprüngliche Preisreihe "
                "zeigt einen klaren Trend, während die Log-Returns stabiler um null "
                "schwanken. Dadurch eignen sie sich besser zur Analyse kurzfristiger "
                "Preisänderungen."
            ),
            plot_image(p["lr"]),
        ),

        # ============================================================
        # 4 · Stationaritätstests
        # ============================================================
        card(
            section_title("4 · Stationaritätsanalyse"),
            results_table([
                (
                    "ADF-Test Originalreihe",
                    f"Statistik = {m['adf_stat']:.3f}, p = {m['adf_p']:.4f}"
                ),
                (
                    "KPSS-Test Originalreihe",
                    f"Statistik = {m['kpss_stat']:.3f}, p = {m['kpss_p']:.4f}"
                ),
                (
                    "ADF-Test nach 1. Differenzierung",
                    f"Statistik = {m['adf_diff_stat']:.3f}, p = {m['adf_diff_p']:.4f}"
                ),
                (
                    "Integrationsordnung",
                    "d = 1"
                ),
            ]),
            info_block(
                "Der ADF-Test der ursprünglichen Goldpreisreihe spricht gegen Stationarität, "
                "während der KPSS-Test die Stationarität ebenfalls ablehnt. Nach der ersten "
                "Differenzierung wird die Reihe stationär. Daher wird für das ARIMA-Modell "
                "die Integrationsordnung d = 1 verwendet."
            ),
            plot_image(p["diff"]),
        ),

        # ============================================================
        # 5 · ACF und PACF
        # ============================================================
        card(
            section_title("5 · Modellidentifikation mit ACF und PACF"),
            info_block(
                "Die ACF- und PACF-Plots der differenzierten Goldpreisreihe zeigen nur "
                "noch kleinere frühe Ausschläge. Das deutet darauf hin, dass einfache "
                "ARIMA-Modelle mit kleinen p- und q-Werten geeignet sind. Für die finale "
                "Analyse wird ein ARIMA(2,1,2)-Modell verwendet."
            ),
            plot_image(p["acf"]),
        ),

        # ============================================================
        # 6 · Modellgüte
        # ============================================================
        card(
            section_title("6 · ARIMA(2,1,2): Modellgüte und Testergebnisse"),
            results_table([
                ("Finales Modell", "ARIMA(2,1,2)"),
                ("AIC", f"{m['aic']:.2f}"),
                ("BIC", f"{m['bic']:.2f}"),
                (
                    "Ljung-Box-Test, Lag 10",
                    f"p = {m['lb_p']:.3f} — keine signifikante Autokorrelation der Residuen"
                ),
                (
                    "Jarque-Bera-Test",
                    f"Statistik = {m['jb_stat']:.2f}, p = {m['jb_p']:.4f}"
                ),
            ]),
            info_block(
                "Das finale ARIMA(2,1,2)-Modell beschreibt die Goldpreis-Zeitreihe solide. "
                "Der Ljung-Box-Test zeigt keine signifikante verbleibende Autokorrelation "
                "in den Residuen. Der Jarque-Bera-Test weist jedoch auf Nicht-Normalität "
                "hin, was bei Finanzzeitreihen aufgrund von Extremwerten und Fat Tails "
                "typisch ist."
            ),
        ),

        # ============================================================
        # 7 · Residualanalyse
        # ============================================================
        card(
            section_title("7 · Residualanalyse"),
            info_block(
                "Die Residuen zeigen kein deutliches systematisches Muster. Dadurch wird "
                "bestätigt, dass das Modell die wichtigsten zeitlichen Abhängigkeiten der "
                "Goldpreisreihe erfasst. Die Verteilung der Residuen ist jedoch nicht "
                "vollständig normalverteilt."
            ),
            plot_image(p["resid"]),
        ),

        # ============================================================
        # 8 · ACF der Residuen
        # ============================================================
        card(
            section_title("8 · Autokorrelation der Residuen"),
            info_block(
                "Die ACF der Residuen dient als zusätzliche Kontrolle. Wenn keine starken "
                "signifikanten Ausschläge mehr sichtbar sind, spricht das dafür, dass das "
                "ARIMA-Modell die zeitlichen Abhängigkeiten ausreichend berücksichtigt."
            ),
            plot_image(p["resid_acf"]),
        ),

        # ============================================================
        # 9 · In-Sample-Fit
        # ============================================================
        card(
            section_title("9 · In-Sample-Fit"),
            info_block(
                "Der In-Sample-Fit vergleicht die tatsächlichen Goldpreise mit den vom "
                "Modell geschätzten Werten. Eine enge Überlagerung beider Linien zeigt, "
                "dass das Modell den historischen Verlauf der Zeitreihe gut nachvollzieht."
            ),
            plot_image(p["fit"]),
        ),

        # ============================================================
        # 10 · Forecast
        # ============================================================
        card(
            section_title("10 · 10-Tage-Forecast"),
            info_block(
                "Der 10-Tage-Forecast zeigt die kurzfristige erwartete Entwicklung des "
                "Goldpreises auf Basis des ARIMA(2,1,2)-Modells. Das 95%-Konfidenzintervall "
                "verdeutlicht die Prognoseunsicherheit, die mit zunehmendem Horizont größer "
                "wird."
            ),
            plot_image(p["fc"]),
        ),

        # ============================================================
        # 11 · Fazit
        # ============================================================
        card(
            section_title("11 · Fazit"),
            info_block(
                "Die Analyse zeigt, dass die ursprüngliche Goldpreis-Zeitreihe nicht "
                "stationär ist und daher differenziert werden muss. Nach der ersten "
                "Differenzierung kann ein ARIMA(2,1,2)-Modell verwendet werden. Das Modell "
                "erfasst die zeitlichen Abhängigkeiten gut und eignet sich für kurzfristige "
                "Prognosen. Gleichzeitig bleiben typische Eigenschaften von Finanzdaten "
                "sichtbar, insbesondere starke Schwankungen, Extremwerte und nicht-normal "
                "verteilte Residuen."
            ),
        ),

    ])
```

---

