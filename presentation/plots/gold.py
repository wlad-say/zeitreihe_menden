"""
plots/gold.py
Berechnet alle Plots, Tests und Modelle für den Gold-Tab.
Modell: ARIMA(2,1,2) auf täglichen Goldpreisen.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from layout.components import fig_to_b64


warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110})

MONTHS_DE = [
    "Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
    "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"
]


def build(df_gold: pd.DataFrame) -> dict:
    """
    Erwartet den aufbereiteten Gold-DataFrame aus data_loader.load_gold().

    Benötigte Spalten:
    - Price

    Falls diese Spalten fehlen, werden sie hier erzeugt:
    - Log_Return
    - Price_diff
    """

    df = df_gold.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"])

    # Falls im DataFrame noch nicht vorhanden, hier erzeugen
    if "Log_Return" not in df.columns:
        df["Log_Return"] = np.log(df["Price"] / df["Price"].shift(1))

    if "Price_diff" not in df.columns:
        df["Price_diff"] = df["Price"].diff()

    df_diff = df.dropna(subset=["Price_diff"])
    df_lr = df.dropna(subset=["Log_Return"])
    price_series = df["Price"].dropna()

    # ============================================================
    # Stationaritätstests
    # ============================================================

    # ADF Originalreihe
    adf_result = adfuller(price_series)
    adf_stat = float(adf_result[0])
    adf_p = float(adf_result[1])

    # KPSS Originalreihe
    kpss_result = kpss(price_series, regression="c", nlags="auto")
    kpss_stat = float(kpss_result[0])
    kpss_p = float(kpss_result[1])

    # ADF differenzierte Reihe
    adf_diff_result = adfuller(df_diff["Price_diff"].dropna())
    adf_diff_stat = float(adf_diff_result[0])
    adf_diff_p = float(adf_diff_result[1])

    # ============================================================
    # 1 · Zeitreihe mit Ereignismarkierungen
    # ============================================================

    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(
        df.index,
        df["Price"],
        color="#b8860b",
        linewidth=0.9,
        label="Goldpreis"
    )

    for label, date, color in [
        ("Eurokrise", "2011-09-01", "#27ae60"),
        ("Corona-Beginn", "2020-03-01", "#c0392b"),
    ]:
        ax.axvline(
            pd.Timestamp(date),
            color=color,
            linestyle="--",
            alpha=0.75,
            label=label
        )

    ax.set_title("Goldpreis-Zeitreihe mit wichtigen Ereignissen", fontweight="bold")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Goldpreis (USD)")
    ax.legend()
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    p_ts = fig_to_b64(fig)

    # ============================================================
    # 2 · EDA: Histogramm + Boxplot nach Monat
    # ============================================================

    df["month_name"] = pd.Categorical(
        df.index.month.map(lambda x: MONTHS_DE[x - 1]),
        categories=MONTHS_DE,
        ordered=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].hist(
        df["Price"].dropna(),
        bins=30,
        color="navy",
        alpha=0.8
    )
    axes[0].set_title("Goldpreis-Verteilung")
    axes[0].set_xlabel("Goldpreis (USD)")
    axes[0].set_ylabel("Häufigkeit")
    axes[0].grid(True, alpha=0.2)

    df.boxplot(
        column="Price",
        by="month_name",
        ax=axes[1],
        boxprops={"color": "navy"},
        medianprops={"color": "#c0392b"},
        whiskerprops={"color": "navy"},
        capprops={"color": "navy"},
        flierprops={
            "marker": "o",
            "color": "navy",
            "alpha": 0.3,
            "markersize": 3,
        },
    )

    axes[1].set_title("Goldpreise nach Monat")
    axes[1].set_xlabel("Monat")
    axes[1].set_ylabel("Goldpreis (USD)")

    fig.suptitle("")
    plt.tight_layout()
    p_eda = fig_to_b64(fig)

    # ============================================================
    # 3 · Original vs. Log-Returns
    # ============================================================

    fig, axes = plt.subplots(2, 1, figsize=(13, 7))

    axes[0].plot(df.index, df["Price"], color="navy")
    axes[0].set_title("Goldpreis (original)", fontweight="bold")
    axes[0].set_ylabel("USD")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(df_lr.index, df_lr["Log_Return"], color="#e07b39")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Log-Returns", fontweight="bold")
    axes[1].set_ylabel("Log-Return")
    axes[1].set_xlabel("Datum")
    axes[1].grid(True, alpha=0.25)

    for a in axes:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    p_lr = fig_to_b64(fig)

    # ============================================================
    # 4 · Differenzierte Goldpreis-Zeitreihe
    # ============================================================

    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(
        df_diff.index,
        df_diff["Price_diff"],
        color="darkred",
        linewidth=0.75
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Differenzierte Goldpreis-Zeitreihe", fontweight="bold")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Differenzierter Goldpreis")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    p_diff = fig_to_b64(fig)

    # ============================================================
    # 5 · ACF/PACF differenzierte Reihe
    # ============================================================

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    plot_acf(df_diff["Price_diff"], ax=axes[0], lags=40)
    axes[0].set_title("ACF – differenzierte Goldpreis-Reihe")

    plot_pacf(df_diff["Price_diff"], ax=axes[1], lags=40, method="ywm")
    axes[1].set_title("PACF – differenzierte Goldpreis-Reihe")

    plt.tight_layout()
    p_acf = fig_to_b64(fig)

    # ============================================================
    # 6 · ARIMA(2,1,2)
    # ============================================================

    model = ARIMA(price_series, order=(2, 1, 2)).fit()
    residuals = model.resid

    aic = float(model.aic)
    bic = float(model.bic)

    # Ljung-Box-Test
    ljung_box = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_p = float(ljung_box["lb_pvalue"].iloc[0])

    # Jarque-Bera-Test
    jb = jarque_bera(residuals)
    jb_stat = float(jb.statistic)
    jb_p = float(jb.pvalue)

    # ============================================================
    # 7 · Residualanalyse
    # ============================================================

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(residuals, color="steelblue", linewidth=0.6)
    axes[0].set_title("Residuen – ARIMA(2,1,2)")
    axes[0].set_xlabel("Zeit")
    axes[0].set_ylabel("Residuen")
    axes[0].grid(True, alpha=0.25)

    axes[1].hist(
        residuals,
        bins=30,
        color="steelblue",
        alpha=0.8,
        edgecolor="white"
    )
    axes[1].set_title("Verteilung der Residuen")
    axes[1].set_xlabel("Residuen")
    axes[1].set_ylabel("Häufigkeit")

    plt.suptitle(
        f"Residualanalyse   Ljung-Box p={lb_p:.3f}   JB p={jb_p:.4f}",
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()
    p_resid = fig_to_b64(fig)

    # ============================================================
    # 8 · ACF der Residuen
    # ============================================================

    fig, ax = plt.subplots(figsize=(13, 4))
    plot_acf(residuals.dropna(), ax=ax, lags=40)
    ax.set_title("ACF der Residuen – ARIMA(2,1,2)", fontweight="bold")
    plt.tight_layout()
    p_resid_acf = fig_to_b64(fig)

    # ============================================================
    # 9 · In-Sample-Fit
    # ============================================================

    fitted_values = model.fittedvalues

    fig, ax = plt.subplots(figsize=(13, 4.5))

    ax.plot(
        price_series.index,
        price_series.values,
        color="#b8860b",
        linewidth=1,
        label="Originale Goldpreise"
    )

    ax.plot(
        fitted_values.index,
        fitted_values.values,
        color="#c0392b",
        linewidth=1,
        label="In-Sample-Fit"
    )

    ax.set_title("In-Sample-Fit des ARIMA(2,1,2)-Modells", fontweight="bold")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Goldpreis (USD)")
    ax.legend()
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    p_fit = fig_to_b64(fig)

    # ============================================================
    # 10 · 10-Tage-Forecast
    # ============================================================

    fc_obj = model.get_forecast(steps=10)
    fc_mean = fc_obj.predicted_mean
    fc_ci = fc_obj.conf_int(alpha=0.05)

    fc_dates = pd.date_range(
        start=price_series.index[-1] + pd.Timedelta(days=1),
        periods=10,
        freq="D"
    )

    fc_mean.index = fc_dates
    fc_ci.index = fc_dates

    hist60 = price_series.iloc[-60:]

    bridge_x = [price_series.index[-1]] + list(fc_dates)
    bridge_y = [price_series.iloc[-1]] + list(fc_mean.values)

    fig, ax = plt.subplots(figsize=(13, 4.5))

    ax.plot(
        hist60.index,
        hist60.values,
        color="#b8860b",
        linewidth=1.5,
        label="Historisch (60 Tage)"
    )

    ax.plot(
        bridge_x,
        bridge_y,
        color="#e07b39",
        linewidth=2,
        linestyle="--",
        label="Forecast"
    )

    ax.fill_between(
        fc_dates,
        fc_ci.iloc[:, 0],
        fc_ci.iloc[:, 1],
        alpha=0.2,
        color="#e07b39",
        label="95%-Konfidenzintervall"
    )

    ax.axvline(
        price_series.index[-1],
        color="gray",
        linewidth=0.8,
        linestyle="--"
    )

    ax.set_title(
        f"ARIMA(2,1,2) – 10-Tage-Forecast Goldpreis   "
        f"Letzter Kurs: ${price_series.iloc[-1]:.0f}",
        fontweight="bold",
    )

    ax.set_ylabel("USD")
    ax.set_xlabel("Datum")
    ax.legend()
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    p_fc = fig_to_b64(fig)

    # ============================================================
    # Rückgabe an layout/tab_gold.py
    # ============================================================

    return {
        "plots": {
            "ts": p_ts,
            "eda": p_eda,
            "lr": p_lr,
            "diff": p_diff,
            "acf": p_acf,
            "resid": p_resid,
            "resid_acf": p_resid_acf,
            "fit": p_fit,
            "fc": p_fc,
        },
        "metrics": {
            "adf_stat": adf_stat,
            "adf_p": adf_p,
            "kpss_stat": kpss_stat,
            "kpss_p": kpss_p,
            "adf_diff_stat": adf_diff_stat,
            "adf_diff_p": adf_diff_p,
            "aic": aic,
            "bic": bic,
            "lb_p": lb_p,
            "jb_p": jb_p,
            "jb_stat": jb_stat,
        },
    }
