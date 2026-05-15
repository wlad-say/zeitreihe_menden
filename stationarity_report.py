import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity_report(series: pd.Series, title: str = "Zeitreihe") -> None:
    """
    Führt ADF- und KPSS-Stationaritätstest auf einer Zeitreihe durch
    und gibt die Ergebnisse formatiert in der Konsole aus.

    Parameter:
        series : pd.Series  – Die zu testende Zeitreihe
        title  : str        – Bezeichnung der Zeitreihe (für die Ausgabe)
    """

    print("=" * 50)
    print(f" Stationaritätstest: {title}")
    print("=" * 50)

    # ------------------------------------------------------------------
    # ADF-Test (Augmented Dickey-Fuller)
    # H0: Einheitswurzel vorhanden → Zeitreihe ist NICHT stationär
    # ------------------------------------------------------------------
    adf_result = adfuller(series.dropna(), autolag="AIC")
    adf_stat   = adf_result[0]
    adf_pval   = adf_result[1]
    adf_lags   = adf_result[2]
    adf_crit   = adf_result[4]

    print("\n ADF Test")
    print(f"   Teststatistik : {adf_stat:.4f}")
    print(f"   p-Wert        : {adf_pval:.4f}")
    print(f"   Lags genutzt  : {adf_lags}")
    print(f"   Krit. Werte   : " + ", ".join([f"{k}={v:.3f}" for k, v in adf_crit.items()]))

    # ------------------------------------------------------------------
    # KPSS-Test (Kwiatkowski–Phillips–Schmidt–Shin)
    # H0: Zeitreihe ist stationär
    # ------------------------------------------------------------------
    kpss_result = kpss(series.dropna(), regression="c", nlags="auto")
    kpss_stat   = kpss_result[0]
    kpss_pval   = kpss_result[1]
    kpss_crit   = kpss_result[3]

    print("\n KPSS Test")
    print(f"   Teststatistik : {kpss_stat:.4f}")
    print(f"   p-Wert        : {kpss_pval:.4f}")
    print(f"   Krit. Werte   : {kpss_crit}")

    print("=" * 50 + "\n")