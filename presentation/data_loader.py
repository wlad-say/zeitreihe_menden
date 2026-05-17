"""
data_loader.py
Lädt alle drei Datensätze über das preprocess-Modul und stellt
sie als saubere DataFrames bereit.

Voraussetzung: preprocess.py liegt im Repo-Root (eine Ebene über /presentation/).
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Repo-Root zum Python-Pfad hinzufügen, damit preprocess importierbar ist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import preprocess


def load_apple() -> pd.DataFrame:
    """
    Gibt den Apple-Datensatz zurück mit zusätzlichen Spalten:
      - Log_Return   : täglicher Log-Return auf Basis des Close-Preises
      - Log_Volume   : log-transformiertes Handelsvolumen
    Index: DatetimeIndex (aufsteigend)
    """
    df = preprocess.process_apple().sort_index()
    df["Log_Return"] = np.log(df["Close"]).diff()
    df["Log_Volume"]  = np.log(df["Volume"])
    df = df.dropna(subset=["Log_Return"])
    return df


def load_weather() -> pd.DataFrame:
    """
    Gibt den Wetter-Datensatz zurück (Tagesdurchschnitte).
    Fehlende Tage werden per Zeitinterpolation geschlossen.
    Spalte: T_mean_degC
    """
    df = preprocess.process_weather()
    df.index = pd.to_datetime(df.index)
    df = df.asfreq("D")
    df["T_mean_degC"] = df["T_mean_degC"].interpolate(method="time")
    return df


def load_gold() -> pd.DataFrame:
    """
    Gibt den Gold-Datensatz zurück mit zusätzlichen Spalten:
      - Price_diff : erste Differenz (für Stationarität)
      - Log_Return : täglicher Log-Return
    Tagesaggregat (resample D, Mittelwert, interpoliert).
    """
    df = preprocess.process_gold()
    if "Date" in df.columns:
        df = df.assign(Date=pd.to_datetime(df["Date"])).set_index("Date")
    df.index = pd.to_datetime(df.index)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Auf Tagesdaten aggregieren
    daily = df["Price"].resample("D").mean().to_frame()
    daily["Price"] = daily["Price"].interpolate()
    daily["Price_diff"] = daily["Price"].diff()
    daily["Log_Return"]  = np.log(daily["Price"]).diff()
    return daily
