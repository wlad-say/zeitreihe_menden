"""
plots/gold.py
Berechnet alle Plots und Modelle für den Gold-Tab.
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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from layout.components import fig_to_b64

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110})

MONTHS_DE = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
             "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]


def build(df_gold: pd.DataFrame) -> dict:
    """
    Erwartet den aufbereiteten Gold-DataFrame aus data_loader.load_gold().
    """
    df = df_gold.copy()
    df_diff = df.dropna(subset=["Price_diff"])

    # ── 1 · Zeitreihe mit Ereignismarkierungen ────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df.index, df["Price"], color="#b8860b", linewidth=0.9, label="Goldpreis")
    for label, date, color in [
        ("Corona-Beginn", "2020-03-01", "#c0392b"),
        ("Eurokrise",     "2011-09-01", "#27ae60"),
    ]:
        ax.axvline(pd.Timestamp(date), color=color, ls="--",
                   alpha=0.75, label=label)
    ax.set_title("Goldpreis-Zeitreihe mit wichtigen Ereignissen", fontweight="bold")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Goldpreis (USD)")
    ax.legend()
    plt.tight_layout()
    p_ts = fig_to_b64(fig)

    # ── 2 · EDA: Histogramm + Boxplot nach Monat ─────────────────────────────
    df["month_name"] = pd.Categorical(
        df.index.month.map(lambda x: MONTHS_DE[x - 1]),
        categories=MONTHS_DE, ordered=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(df["Price"].dropna(), bins=30, color="navy", alpha=0.8)
    axes[0].set(title="Goldpreis-Verteilung",
                xlabel="Goldpreis (USD)", ylabel="Häufigkeit")
    df.boxplot(
        column="Price", by="month_name", ax=axes[1],
        boxprops={"color": "navy"}, medianprops={"color": "#c0392b"},
        whiskerprops={"color": "navy"}, capprops={"color": "navy"},
        flierprops={"marker": "o", "color": "navy", "alpha": 0.3, "markersize": 3},
    )
    axes[1].set(title="Goldpreise nach Monat",
                xlabel="Monat", ylabel="Goldpreis (USD)")
    fig.suptitle("")
    plt.tight_layout()
    p_eda = fig_to_b64(fig)

    # ── 3 · Original vs. Log-Returns ─────────────────────────────────────────
    df_lr = df.dropna(subset=["Log_Return"])
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    axes[0].plot(df.index, df["Price"], color="navy")
    axes[0].set_title("Goldpreis (original)", fontweight="bold")
    axes[0].set_ylabel("USD")
    axes[1].plot(df_lr.index, df_lr["Log_Return"], color="#e07b39")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Log-Returns", fontweight="bold")
    axes[1].set_ylabel("Log-Return")
    for a in axes:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    p_lr = fig_to_b64(fig)

    # ── 4 · ACF/PACF differenzierte Reihe ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_acf(df_diff["Price_diff"], ax=axes[0], lags=40)
    axes[0].set_title("ACF – differenzierte Goldpreis-Reihe")
    plot_pacf(df_diff["Price_diff"], ax=axes[1], lags=40)
    axes[1].set_title("PACF – differenzierte Goldpreis-Reihe")
    plt.tight_layout()
    p_acf = fig_to_b64(fig)

    # ── Modell: ARIMA(2,1,2) ─────────────────────────────────────────────────
    price_series = df["Price"].dropna()
    model = ARIMA(price_series, order=(2, 1, 2)).fit()
    residuals = model.resid

    lb_p  = acorr_ljungbox(residuals, lags=[10], return_df=True)["lb_pvalue"].values[0]
    jb_stat, jb_p = jarque_bera(residuals)

    # ── 5 · Residualanalyse ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(residuals, color="steelblue", linewidth=0.6)
    axes[0].set_title("Residuen – ARIMA(2,1,2)")
    axes[0].set_ylabel("Residuen")
    axes[1].hist(residuals, bins=30, color="steelblue", alpha=0.8, edgecolor="white")
    axes[1].set_title("Verteilung der Residuen")
    axes[1].set_xlabel("Residuen")
    plt.suptitle(
        f"Residualanalyse   Ljung-Box p={lb_p:.3f}   JB p={jb_p:.4f}",
        fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    p_resid = fig_to_b64(fig)

    # ── 6 · 10-Tage-Forecast ─────────────────────────────────────────────────
    fc_obj    = model.get_forecast(steps=10)
    fc_mean   = fc_obj.predicted_mean
    fc_ci     = fc_obj.conf_int(alpha=0.05)
    fc_dates  = pd.date_range(
        start=price_series.index[-1] + pd.Timedelta(days=1), periods=10, freq="D"
    )
    fc_mean.index = fc_ci.index = fc_dates

    hist60   = price_series.iloc[-60:]
    bridge_x = [price_series.index[-1]] + list(fc_dates)
    bridge_y = [price_series.iloc[-1]] + list(fc_mean.values)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(hist60.index, hist60.values,
            color="#b8860b", linewidth=1.5, label="Historisch (60 Tage)")
    ax.plot(bridge_x, bridge_y,
            color="#e07b39", linewidth=2, linestyle="--", label="Forecast")
    ax.fill_between(fc_dates, fc_ci.iloc[:, 0], fc_ci.iloc[:, 1],
                    alpha=0.2, color="#e07b39", label="95%-Konfidenzintervall")
    ax.axvline(price_series.index[-1], color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(
        f"ARIMA(2,1,2) – 10-Tage-Forecast Goldpreis   "
        f"Letzter Kurs: ${price_series.iloc[-1]:.0f}",
        fontweight="bold",
    )
    ax.set_ylabel("USD")
    ax.set_xlabel("Datum")
    ax.legend()
    plt.tight_layout()
    p_fc = fig_to_b64(fig)

    return {
        "plots": {
            "ts":    p_ts,
            "eda":   p_eda,
            "lr":    p_lr,
            "acf":   p_acf,
            "resid": p_resid,
            "fc":    p_fc,
        },
        "metrics": {
            "lb_p":   lb_p,
            "jb_p":   jb_p,
            "jb_stat": jb_stat,
        },
    }
