"""
plots/var_model.py
Berechnet alle Plots und Modelle für den VAR-Tab (Multivariate Zeitreihenanalyse).
Variablen: Apple Close Log-Return, Gold Log-Return, Delta Temperatur.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from layout.components import fig_to_b64

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110})

COLORS = {
    "apple":   "steelblue",
    "gold":    "#b8860b",
    "weather": "#2e8b57",
    "fc":      "#e07b39",
}


def build(df_apple: pd.DataFrame,
          df_gold: pd.DataFrame,
          df_weather: pd.DataFrame) -> dict:
    """
    Erwartet die aufbereiteten DataFrames aus data_loader.
    Baut einen gemeinsamen Datensatz und schätzt ein VAR-Modell.
    """
    # ── Gemeinsamen Datensatz zusammenstellen ─────────────────────────────────
    apple_lr  = df_apple["Log_Return"].resample("D").last()
    gold_lr   = np.log(df_gold["Price"]).diff().resample("D").last()
    temp_diff = df_weather["T_mean_degC"].diff()

    df_var = pd.concat(
        [apple_lr.rename("Apple_LR"),
         gold_lr.rename("Gold_LR"),
         temp_diff.rename("Temp_diff")],
        axis=1, join="inner",
    ).dropna()
    df_var = df_var[df_var.index.year >= 2010]

    # ── 1 · Transformierte Zeitreihen ─────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for ax, col, color, label in zip(
        axes,
        ["Apple_LR", "Gold_LR", "Temp_diff"],
        [COLORS["apple"], COLORS["gold"], COLORS["weather"]],
        ["Apple Log-Return", "Gold Log-Return", "Temperatur Δ (°C)"],
    ):
        ax.plot(df_var.index, df_var[col], color=color, linewidth=0.45)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_ylabel(label, fontsize=9)
    axes[0].set_title(
        "Transformierte Zeitreihen – stationär (ab 2010)", fontweight="bold"
    )
    axes[2].set_xlabel("Datum")
    plt.tight_layout()
    p_ts = fig_to_b64(fig)

    # ── 2 · Korrelationsmatrix ────────────────────────────────────────────────
    corr = df_var.corr()
    labels = ["Apple LR", "Gold LR", "Δ Temp"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr.values[i, j]:.3f}",
                    ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_title("Korrelationsmatrix der transformierten Reihen", fontweight="bold")
    plt.tight_layout()
    p_corr = fig_to_b64(fig)

    # ── VAR-Modell schätzen ───────────────────────────────────────────────────
    split = int(len(df_var) * 0.9)
    train_v = df_var.iloc[:split]

    var_mdl    = VAR(train_v)
    lag_result = var_mdl.select_order(maxlags=10)
    # select_order kann 0 zurückgeben → Minimum 2 erzwingen
    aic_lag  = lag_result.aic
    best_lag = max(2, int(aic_lag)) if (aic_lag is not None and aic_lag > 0) else 2
    var_fit  = var_mdl.fit(best_lag)

    # ── 3 · Impuls-Antwort-Funktionen ────────────────────────────────────────
    try:
        irf = var_fit.irf(10)
        fig = irf.plot(orth=True, figsize=(13, 9))
        fig.suptitle(
            "Impuls-Antwort-Funktionen (IRF) – orthogonalisiert",
            fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        p_irf = fig_to_b64(fig)
    except Exception:
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.text(0.5, 0.5, "IRF-Berechnung nicht verfügbar",
                ha="center", va="center", fontsize=14, color="gray")
        ax.axis("off")
        p_irf = fig_to_b64(fig)

    # ── 4 · FEVD ──────────────────────────────────────────────────────────────
    try:
        fevd = var_fit.fevd(10)
        fig  = fevd.plot(figsize=(13, 6))
        fig.suptitle(
            "Forecast Error Variance Decomposition (FEVD)",
            fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        p_fevd = fig_to_b64(fig)
    except Exception:
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.text(0.5, 0.5, "FEVD-Berechnung nicht verfügbar",
                ha="center", va="center", fontsize=14, color="gray")
        ax.axis("off")
        p_fevd = fig_to_b64(fig)

    # ── 5 · Residuen ──────────────────────────────────────────────────────────
    resid = var_fit.resid
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for ax, col, color, label in zip(
        axes,
        ["Apple_LR", "Gold_LR", "Temp_diff"],
        [COLORS["apple"], COLORS["gold"], COLORS["weather"]],
        ["Apple LR", "Gold LR", "Δ Temp"],
    ):
        ax.plot(resid.index, resid[col], color=color, linewidth=0.45)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_ylabel(label, fontsize=9)
    axes[0].set_title("VAR-Modell – Residuen", fontweight="bold")
    axes[2].set_xlabel("Datum")
    plt.tight_layout()
    p_resid = fig_to_b64(fig)

    # ── 6 · 10-Tage-Forecast ─────────────────────────────────────────────────
    k_ar   = max(var_fit.k_ar, 1)
    fc_in  = df_var.values[-k_ar:]
    fc_out = var_fit.forecast(y=fc_in, steps=10)
    fc_df  = pd.DataFrame(fc_out, columns=df_var.columns)
    fc_df.index = pd.date_range(
        start=df_var.index[-1] + pd.Timedelta(days=1), periods=10, freq="D"
    )

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    hist_n = 60
    for ax, col, color, label in zip(
        axes,
        ["Apple_LR", "Gold_LR", "Temp_diff"],
        [COLORS["apple"], COLORS["gold"], COLORS["weather"]],
        ["Apple Log-Return", "Gold Log-Return", "Temperatur Δ (°C)"],
    ):
        hist = df_var[col].iloc[-hist_n:]
        ax.plot(hist.index, hist, color=color, linewidth=0.8, label="Historisch")
        bx = [df_var.index[-1]] + list(fc_df.index)
        by = [df_var[col].iloc[-1]] + list(fc_df[col].values)
        ax.plot(bx, by, color=COLORS["fc"], linewidth=2,
                linestyle="--", label="Forecast")
        ax.set_ylabel(label, fontsize=9)
    axes[0].set_title(
        f"VAR({best_lag})-Modell – 10-Tage-Forecast", fontweight="bold"
    )
    axes[0].legend(fontsize=8)
    axes[2].set_xlabel("Datum")
    plt.tight_layout()
    p_fc = fig_to_b64(fig)

    return {
        "plots": {
            "ts":    p_ts,
            "corr":  p_corr,
            "irf":   p_irf,
            "fevd":  p_fevd,
            "resid": p_resid,
            "fc":    p_fc,
        },
        "metrics": {
            "best_lag": best_lag,
            "n_obs":    len(df_var),
        },
    }
