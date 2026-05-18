"""
preprocess.py
Datenbereinigung für Gold Price, Apple Stock und Wetterdaten.

Verwendung im Haupt-Notebook:
    import preprocess
    preprocess.run()

Oder einzelne Datensätze:
    preprocess.process_gold()
    preprocess.process_apple()
    preprocess.process_weather()

Verzeichnisstruktur:
    data/
    ├── raw/
    │   ├── GoldPrice.csv
    │   ├── apple_stock.csv
    │   └── weather_ts.csv
    └── processed/
        ├── GoldPrice_processed.csv
        ├── apple_stock_processed.csv
        └── weather_processed.csv
"""

import pandas as pd
from pathlib import Path

# ── Pfade ──────────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def _ensure_dirs():
    """Stellt sicher, dass der processed-Ordner existiert."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ── Gold Price ─────────────────────────────────────────────────────────────────
def process_gold(filename: str = "GoldPrice.csv") -> pd.DataFrame:
    """
    Bereinigt den Gold-Datensatz.
    Alle Spalten bleiben erhalten; Datum wird geparst und als Index gesetzt.
    """
    df = pd.read_csv(RAW_DIR / filename)

    # Datum parsen, sortieren und als Index setzen
    df["Date"] = pd.to_datetime(df["Date"], format="%b %d, %Y")
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.set_index("Date")

    out_path = PROCESSED_DIR / "GoldPrice_processed.csv"
    df.to_csv(out_path)
    print(f"[Gold]  Gespeichert → {out_path}")
    return df


# ── Apple Stock ────────────────────────────────────────────────────────────────
def process_apple(filename: str = "apple_stock.csv") -> pd.DataFrame:
    """
    Bereinigt den Apple-Stock-Datensatz.
    Alle Spalten bleiben erhalten; Datum wird geparst und als Index gesetzt.
 
    CSV-Struktur:
        Zeile 1: Price, Close, High, Low, Open, Volume  ← echte Spaltennamen
        Zeile 2: Ticker, AAPL, AAPL, ...               ← überspringen
        Zeile 3: Date, , , , ,                          ← überspringen
        Zeile 4+: Daten
    """
    # Spaltennamen aus Zeile 1 lesen
    col_names = pd.read_csv(RAW_DIR / filename, nrows=0).columns.tolist()
    # Erste Spalte manuell auf "Date" setzen
    col_names[0] = "Date"
 
    # Daten ab Zeile 4 einlesen (skiprows überspringt Zeilen 1-3)
    df = pd.read_csv(RAW_DIR / filename, skiprows=3, header=None, names=col_names)
 
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.set_index("Date")
 
    out_path = PROCESSED_DIR / "apple_stock_processed.csv"
    df.to_csv(out_path)
    print(f"[Apple]  Gespeichert → {out_path}")
    return df


# ── Weather ────────────────────────────────────────────────────────────────────
def process_weather(filename: str = "weather_ts.csv") -> pd.DataFrame:
    """
    Bereinigt den Wetterdatensatz (10-Minuten → täglich).
    Behält: Date, T_mean (Tagesdurchschnitt der Temperatur in °C)
    Aggregation: arithmetischer Mittelwert aller 10-min-Messungen pro Tag
    """
    df = pd.read_csv(RAW_DIR / filename)

    # Datum/Zeit parsen
    df["Date Time"] = pd.to_datetime(df["Date Time"], dayfirst=True)
    df["Date"] = df["Date Time"].dt.normalize()   # auf Tagesebene kürzen

    # Täglichen Temperaturmittelwert berechnen
    daily = (
        df.groupby("Date")["T (degC)"]
        .mean()
        .round(2)
        .rename("T_mean_degC")
        .to_frame()
    )

    out_path = PROCESSED_DIR / "weather_processed.csv"
    daily.to_csv(out_path)
    print(f"[Weather] Gespeichert → {out_path}")
    return daily


# ── Einstiegspunkt ─────────────────────────────────────────────────────────────
def run():
    """Verarbeitet alle drei Datensätze auf einmal."""
    print("  Datenvorverarbeitung gestartet")
    _ensure_dirs()
    gold    = process_gold()
    apple   = process_apple()
    weather = process_weather()
    print("  Alle Datensätze erfolgreich verarbeitet.")
    return gold, apple, weather


if __name__ == "__main__":
    run()