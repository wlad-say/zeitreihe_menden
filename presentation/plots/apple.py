"""
plots/apple.py
Berechnet alle Plots und Modelle für den Apple-Stock-Tab.
Gibt ein Dict mit Base64-PNG-Strings und Metriken zurück.
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
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from layout.components import fig_to_b64

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi":        110,
    "figure.facecolor":  "#f5f6f8",
    "axes.facecolor":    "#f5f6f8",
    "axes.edgecolor":    "#b0bac5",
    "axes.linewidth":    0.8,
    "grid.color":        "#dde2e8",
    "grid.linewidth":    0.5,
    "xtick.color":       "#5a6878",
    "ytick.color":       "#5a6878",
    "text.color":        "#1e2b3a",
    "axes.labelcolor":   "#1e2b3a",
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.framealpha": 0.85,
    "figure.figsize":    (13, 4.5),
})


def build(df_apple: pd.DataFrame) -> dict:
    """
    Erwartet den aufbereiteten Apple-DataFrame aus data_loader.load_apple().
    Gibt zurück: dict mit Schlüsseln 'plots' (b64-Strings) und 'metrics'.
    """
    df = df_apple.copy()

    # ── 1 · Close-Preis ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df["Close"], color="steelblue", linewidth=1)
    ax.set_title("Apple Aktienkurs (Close-Preis)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Preis (USD)")
    ax.set_xlabel("Datum")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    p_close = fig_to_b64(fig)

    # ── 2 · Log-Returns ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df["Log_Return"], color="darkorange", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Log-Returns (transformiert — stationär)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Log-Return")
    ax.set_xlabel("Datum")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    p_lr = fig_to_b64(fig)

    # ── 3 · ACF & PACF ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(df["Log_Return"], lags=40, ax=axes[0], color="steelblue")
    axes[0].set_title("ACF der Log-Returns", fontweight="bold")
    axes[0].set_xlabel("Lag")
    plot_pacf(df["Log_Return"], lags=40, ax=axes[1], color="darkorange", method="ywm")
    axes[1].set_title("PACF der Log-Returns", fontweight="bold")
    axes[1].set_xlabel("Lag")
    plt.tight_layout()
    p_acf = fig_to_b64(fig)

    # ── Modell: ARIMA(2,0,2) ─────────────────────────────────────────────────
    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test  = df.iloc[split:]

    model = ARIMA(train["Log_Return"], order=(2, 0, 2)).fit()
    residuals = model.resid

    # ── 4 · Residualanalyse ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].plot(residuals, color="steelblue", linewidth=0.8)
    axes[0, 0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Residuen über Zeit – ARIMA(2,0,2)", fontweight="bold")
    axes[0, 1].hist(residuals, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    axes[0, 1].set_title("Histogramm der Residuen", fontweight="bold")
    plot_acf(residuals, lags=40, ax=axes[1, 0], color="steelblue")
    axes[1, 0].set_title("ACF der Residuen (reoptimiert)", fontweight="bold")
    plot_pacf(residuals, lags=40, ax=axes[1, 1], color="darkorange", method="ywm")
    axes[1, 1].set_title("PACF der Residuen (reoptimiert)", fontweight="bold")
    plt.suptitle("Residualanalyse – ARIMA(2,0,2) (reoptimiert)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    p_resid = fig_to_b64(fig)

    # ── 5 · Testset-Evaluation ────────────────────────────────────────────────
    preds = model.forecast(steps=len(test))
    preds.index = test.index
    mae  = mean_absolute_error(test["Log_Return"], preds)
    rmse = np.sqrt(mean_squared_error(test["Log_Return"], preds))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(test.index, test["Log_Return"],
            color="steelblue", linewidth=0.8, alpha=0.85, label="Tatsächlich")
    ax.plot(preds.index, preds,
            color="darkorange", linewidth=1.5, linestyle="--",
            alpha=0.9, label="ARIMA(2,0,2) Prognose")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.axvline(pd.Timestamp("2025-04-07"), color="red", linewidth=1.2,
               linestyle="--", alpha=0.7, label="Trump-Zollschock (Apr 2025)")
    ax.set_title(
        "Testset – Tatsächliche vs. vorhergesagte Log-Returns\n"
        "(Flache Prognose ≈ 0 ist korrekt: Aktienkurse folgen einem Random Walk)",
        fontsize=12, pad=12,
    )
    ax.set_ylabel("Log-Return")
    ax.set_xlabel("Datum")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    plt.tight_layout()
    p_test = fig_to_b64(fig)

    # ── 6 · 10-Tage-Forecast ─────────────────────────────────────────────────
    full_model  = ARIMA(df["Log_Return"], order=(2, 0, 2)).fit()
    fc_obj      = full_model.get_forecast(steps=10)
    fc_mean     = fc_obj.predicted_mean
    fc_ci       = fc_obj.conf_int(alpha=0.05)

    fc_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=10)
    fc_mean.index = fc_ci.index = fc_dates

    # Rückrechnung Log-Returns → Kurspreise
    letzter = df["Close"].iloc[-1]
    def lr2price(lr_vals, start):
        return start * np.exp(np.cumsum(lr_vals))

    fc_preise = lr2price(fc_mean.values, letzter)
    ci_lo     = lr2price(fc_ci.iloc[:, 0].values, letzter)
    ci_hi     = lr2price(fc_ci.iloc[:, 1].values, letzter)
    hist60    = df["Close"].iloc[-60:]

    bridge_x = pd.DatetimeIndex([df.index[-1]]).append(fc_dates)
    bridge_y = np.concatenate([[letzter], fc_preise])
    ci_lo_b  = np.concatenate([[letzter], ci_lo])
    ci_hi_b  = np.concatenate([[letzter], ci_hi])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hist60.index, hist60.values,
            color="steelblue", linewidth=1.5,
            label="Historischer Kurs (letzte 60 Tage)")
    ax.axvline(df.index[-1], color="gray", linewidth=0.8, linestyle="--",
               alpha=0.7, label="Heute")
    ax.fill_between(bridge_x, ci_lo_b, ci_hi_b,
                    alpha=0.25, color="darkorange", label="95%-Konfidenzintervall")
    ax.plot(bridge_x, bridge_y,
            color="darkorange", linewidth=1.8, linestyle="--", label="Prognose (Mittelpfad)")
    ax.plot(df.index[-1], letzter, "o", color="steelblue", markersize=6, zorder=5)
    ax.set_title(
        "Apple-Kurs – 10-Tage-Prognose mit 95%-Konfidenzintervall\n"
        "(ARIMA(2,0,2) — trainiert auf allen verfügbaren Daten)",
        fontsize=13, pad=12,
    )
    ax.set_ylabel("Kurs (USD)")
    ax.set_xlabel("Datum")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=30)
    plt.tight_layout()
    p_fc = fig_to_b64(fig)

    return {
        "plots": {
            "close":  p_close,
            "lr":     p_lr,
            "acf":    p_acf,
            "resid":  p_resid,
            "test":   p_test,
            "fc":     p_fc,
        },
        "metrics": {
            "mae":          mae,
            "rmse":         rmse,
            "letzter_kurs": letzter,
            "fc_kurs":      fc_preise[-1],
            "ci_lo":        ci_lo[-1],
            "ci_hi":        ci_hi[-1],
        },
    }