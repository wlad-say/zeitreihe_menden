"""
plots/weather.py
Berechnet alle Plots und Modelle für den Weather-Tab.
Modell: SARIMAX(3,0,0) mit Fourier-Terms (K=3, Periode=365.25).

Neu (v2):
  - Koeffiziententabelle (AR-Parameter + p-Werte)
  - Walk-Forward Validation (6 Folds, ARIMA vs. Prophet vs. Seasonal Naive)
  - Walk-Forward Boxplots (RMSE & MAE)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import probplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from layout.components import fig_to_b64

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110})

MONTHS_DE = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
             "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]


def _fourier(index: pd.DatetimeIndex, period: float, K: int) -> pd.DataFrame:
    t = np.arange(len(index))
    cols = {}
    for k in range(1, K + 1):
        cols[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=index)


def build(df_weather: pd.DataFrame) -> dict:
    """
    Erwartet den aufbereiteten Weather-DataFrame aus data_loader.load_weather().
    """
    df = df_weather.copy()
    series = df["T_mean_degC"]

    # ── 1 · Zeitreihe ────────────────────────────────────────────────────────
    weekly = series.resample("W").mean()
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df.index, series, color="steelblue", linewidth=0.3, alpha=0.35)
    ax.plot(weekly.index, weekly.values, color="steelblue", linewidth=1.6)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(
        "Temperaturverlauf 2009–2017  (dünn = täglich, dick = Wochenmittel)",
        fontweight="bold",
    )
    ax.set_ylabel("Temperatur (°C)")
    ax.set_xlabel("Datum")
    plt.tight_layout()
    p_ts = fig_to_b64(fig)

    # ── 2 · EDA: Boxplot + Histogramm ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    df.groupby(df.index.month)["T_mean_degC"].apply(list).apply(
        pd.Series
    ).T.boxplot(ax=axes[0])
    axes[0].set_xticklabels(MONTHS_DE)
    axes[0].set_title("Temperaturverteilung nach Monat")
    axes[0].set_ylabel("°C")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=0.8)
    series.plot(kind="hist", bins=40, ax=axes[1],
                color="steelblue", edgecolor="white")
    axes[1].set_title("Histogramm Tagesmitteltemperatur")
    axes[1].set_xlabel("°C")
    plt.tight_layout()
    p_eda = fig_to_b64(fig)

    # ── 3 · ACF & PACF ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    plot_acf(series, lags=50, ax=axes[0], color="steelblue")
    axes[0].set_title("ACF – Kurzfristig (50 Lags)")
    plot_acf(series, lags=400, ax=axes[1], color="steelblue")
    axes[1].axvline(365, color="red", linestyle="--", linewidth=1, label="Lag 365")
    axes[1].legend()
    axes[1].set_title("ACF – Jahressaisonalität (400 Lags)")
    plot_pacf(series, lags=50, ax=axes[2], color="steelblue", method="ywm")
    axes[2].set_title("PACF (50 Lags)")
    plt.tight_layout()
    p_acf = fig_to_b64(fig)

    # ── Modell: SARIMAX(3,0,0) + Fourier ─────────────────────────────────────
    exog_full = _fourier(df.index, period=365.25, K=3)
    train_s   = series.iloc[:-10]
    test_s    = series.iloc[-10:]
    exog_tr   = exog_full.iloc[:-10]
    exog_te   = exog_full.iloc[-10:]

    model = SARIMAX(train_s, order=(3, 0, 0), exog=exog_tr,
                    trend="c", enforce_stationarity=False).fit(disp=False)
    residuals = model.resid
    lb_p = acorr_ljungbox(residuals, lags=[30], return_df=True)["lb_pvalue"].values[0]

    # ── NEU: Koeffizienten extrahieren ────────────────────────────────────────
    params  = model.params
    pvalues = model.pvalues
    ar_rows = []
    for lag in [1, 2, 3]:
        key = f"ar.L{lag}"
        if key in params.index:
            ar_rows.append((
                f"AR(L{lag})",
                f"{params[key]:+.4f}",
                f"{pvalues[key]:.4f}",
                "✓" if pvalues[key] < 0.05 else "✗",
            ))

    # ── 4 · Residualanalyse ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    residuals.plot(ax=axes[0, 0], color="steelblue")
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Residuen über Zeit")
    residuals.plot(kind="hist", bins=40, ax=axes[0, 1],
                   color="steelblue", edgecolor="white")
    axes[0, 1].set_title("Verteilung der Residuen")
    plot_acf(residuals, lags=40, ax=axes[1, 0], color="steelblue")
    axes[1, 0].set_title("ACF der Residuen")
    probplot(residuals, plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot der Residuen")
    plt.suptitle(
        f"Residualanalyse – ARIMA(3,0,0)+Fourier   Ljung-Box p={lb_p:.3f}",
        fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    p_resid = fig_to_b64(fig)

    # ── 5 · Testset-Evaluation ────────────────────────────────────────────────
    fc_test    = model.forecast(steps=10, exog=exog_te)
    fc_test_ci = model.get_forecast(steps=10, exog=exog_te).conf_int(alpha=0.05)
    mae  = mean_absolute_error(test_s, fc_test)
    rmse = np.sqrt(mean_squared_error(test_s, fc_test))

    last_tr = train_s.iloc[[-1]]
    bridge_fc = pd.concat([last_tr, fc_test])
    last_ci   = pd.DataFrame(
        [[last_tr.values[0], last_tr.values[0]]],
        index=last_tr.index, columns=fc_test_ci.columns,
    )
    bridge_ci = pd.concat([last_ci, fc_test_ci])

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(train_s.iloc[-90:].index, train_s.iloc[-90:],
            color="steelblue", linewidth=1.2, label="Training (90 Tage)")
    ax.plot(test_s.index, test_s,
            color="black", linewidth=1.5, linestyle="--", label="Tatsächlich")
    ax.plot(bridge_fc.index, bridge_fc,
            color="#e07b39", linewidth=2, label="Forecast ARIMA(3,0,0)")
    ax.fill_between(bridge_ci.index, bridge_ci.iloc[:, 0], bridge_ci.iloc[:, 1],
                    alpha=0.2, color="#e07b39", label="95%-Konfidenzintervall")
    ax.axvline(train_s.index[-1], color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(
        f"Testset-Evaluation – ARIMA(3,0,0)+Fourier   RMSE = {rmse:.2f} °C   MAE = {mae:.2f} °C",
        fontweight="bold",
    )
    ax.set_ylabel("Temperatur (°C)")
    ax.set_xlabel("Datum")
    ax.legend()
    plt.tight_layout()
    p_test = fig_to_b64(fig)

    # ── 6 · 10-Tage-Forecast ─────────────────────────────────────────────────
    final = SARIMAX(series, order=(3, 0, 0), exog=exog_full,
                    trend="c", enforce_stationarity=False).fit(disp=False)
    fc_idx   = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1), periods=10, freq="D"
    )
    exog_fut = _fourier(fc_idx, period=365.25, K=3)
    fc_obj   = final.get_forecast(steps=10, exog=exog_fut)
    fc_mean  = fc_obj.predicted_mean
    fc_ci    = fc_obj.conf_int(alpha=0.05)
    hist90   = series.iloc[-90:]
    bridge_x = [df.index[-1]] + list(fc_mean.index)
    bridge_v = [series.iloc[-1]] + list(fc_mean.values)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(hist90.index, hist90, color="steelblue", label="Historisch (90 Tage)")
    ax.plot(bridge_x, bridge_v, color="#e07b39", linewidth=2, label="Forecast (10 Tage)")
    ax.fill_between(fc_mean.index, fc_ci.iloc[:, 0], fc_ci.iloc[:, 1],
                    alpha=0.25, color="#e07b39", label="95%-Konfidenzintervall")
    ax.axvline(df.index[-1], color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("ARIMA(3,0,0)+Fourier – 10-Tage-Forecast", fontweight="bold")
    ax.set_ylabel("Temperatur (°C)")
    ax.set_xlabel("Datum")
    ax.legend()
    plt.tight_layout()
    p_fc = fig_to_b64(fig)

    # ── 7 · NEU: Walk-Forward Validation (6 Folds) ───────────────────────────
    H       = 10
    n_folds = 6
    step    = 60
    fold_test_ends = sorted([len(series) - i * step for i in range(n_folds)])

    wf_results = []
    for fold_idx, test_end_idx in enumerate(fold_test_ends, start=1):
        train_end_idx = test_end_idx - H
        tr_sl = series.iloc[:train_end_idx]
        te_sl = series.iloc[train_end_idx:test_end_idx]

        exog_tr_sl = _fourier(tr_sl.index, period=365.25, K=3)
        exog_te_sl = _fourier(te_sl.index, period=365.25, K=3)

        # ARIMA
        m_arima  = SARIMAX(tr_sl, order=(3, 0, 0), exog=exog_tr_sl,
                           trend="c", enforce_stationarity=False).fit(disp=False)
        fc_arima = m_arima.forecast(steps=H, exog=exog_te_sl)
        rmse_arima = np.sqrt(mean_squared_error(te_sl, fc_arima))
        mae_arima  = mean_absolute_error(te_sl, fc_arima)

        # Seasonal Naive (Lag 365)
        fc_naive   = series.iloc[train_end_idx - 365:test_end_idx - 365].values
        rmse_naive = np.sqrt(mean_squared_error(te_sl.values, fc_naive))
        mae_naive  = mean_absolute_error(te_sl.values, fc_naive)

        # Prophet (optional – nur wenn installiert)
        try:
            from prophet import Prophet
            tr_prop = tr_sl.reset_index().rename(
                columns={tr_sl.index.name or "index": "ds", "T_mean_degC": "y"}
            )
            m_prop  = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                              daily_seasonality=False, interval_width=0.95)
            m_prop.fit(tr_prop)
            future   = m_prop.make_future_dataframe(periods=H)
            fc_prop  = m_prop.predict(future).tail(H)["yhat"].values
            rmse_prop = np.sqrt(mean_squared_error(te_sl.values, fc_prop))
            mae_prop  = mean_absolute_error(te_sl.values, fc_prop)
        except Exception:
            rmse_prop = np.nan
            mae_prop  = np.nan

        wf_results.append({
            "Fold":         fold_idx,
            "Test-Start":   te_sl.index.min().date(),
            "Test-Ende":    te_sl.index.max().date(),
            "ARIMA_RMSE":   round(rmse_arima, 3),
            "Prophet_RMSE": round(rmse_prop, 3) if not np.isnan(rmse_prop) else None,
            "SNaive_RMSE":  round(rmse_naive, 3),
            "ARIMA_MAE":    round(mae_arima, 3),
            "Prophet_MAE":  round(mae_prop, 3) if not np.isnan(mae_prop) else None,
            "SNaive_MAE":   round(mae_naive, 3),
        })

    wf_df = pd.DataFrame(wf_results)

    # Walk-Forward Mittelwerte (für results_table im Tab)
    wf_means = {
        "arima_rmse":  round(wf_df["ARIMA_RMSE"].mean(), 3),
        "arima_mae":   round(wf_df["ARIMA_MAE"].mean(), 3),
        "naive_rmse":  round(wf_df["SNaive_RMSE"].mean(), 3),
        "naive_mae":   round(wf_df["SNaive_MAE"].mean(), 3),
    }
    prophet_available = wf_df["Prophet_RMSE"].notna().all()
    if prophet_available:
        wf_means["prophet_rmse"] = round(wf_df["Prophet_RMSE"].mean(), 3)
        wf_means["prophet_mae"]  = round(wf_df["Prophet_MAE"].mean(), 3)

    # Walk-Forward Boxplots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rmse_cols = ["ARIMA_RMSE", "SNaive_RMSE"]
    mae_cols  = ["ARIMA_MAE",  "SNaive_MAE"]
    rmse_labels = ["ARIMA(3,0,0)\n+Fourier", "Seasonal Naive"]
    mae_labels  = ["ARIMA(3,0,0)\n+Fourier", "Seasonal Naive"]

    if prophet_available:
        rmse_cols.insert(1, "Prophet_RMSE")
        mae_cols.insert(1, "Prophet_MAE")
        rmse_labels.insert(1, "Prophet")
        mae_labels.insert(1, "Prophet")

    wf_df[rmse_cols].boxplot(ax=axes[0])
    axes[0].set_xticklabels(rmse_labels)
    axes[0].set_ylabel("RMSE (°C)")
    axes[0].set_title("Walk-Forward Validation – RMSE (6 Folds)")
    axes[0].grid(axis="y", alpha=0.3)

    wf_df[mae_cols].boxplot(ax=axes[1])
    axes[1].set_xticklabels(mae_labels)
    axes[1].set_ylabel("MAE (°C)")
    axes[1].set_title("Walk-Forward Validation – MAE (6 Folds)")
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle(
        "ARIMA(3,0,0)+Fourier vs. Vergleichsmodelle — 6 Expanding-Window-Folds",
        fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    p_wf = fig_to_b64(fig)

    return {
        "plots": {
            "ts":    p_ts,
            "eda":   p_eda,
            "acf":   p_acf,
            "resid": p_resid,
            "test":  p_test,
            "fc":    p_fc,
            "wf":    p_wf,          # NEU
        },
        "metrics": {
            "mae":              mae,
            "rmse":             rmse,
            "lb_p":             lb_p,
            "ar_rows":          ar_rows,          # NEU: [(label, coef, pval, sig), ...]
            "wf_means":         wf_means,          # NEU: Mittelwerte Walk-Forward
            "prophet_available": prophet_available, # NEU
        },
    }