# Zeitreihenanalyse – Präsentations-App

Interaktive Dash-App als Alternative zur klassischen PowerPoint-Präsentation.
Zeigt alle Analyseschritte, Plots und Modellergebnisse aus `ts_menden_team03.ipynb`
in einer strukturierten 4-Tab-Oberfläche.

## Voraussetzungen

- Alle Abhängigkeiten aus `requirements.txt` des Repo-Root installiert
- `preprocess.py` und `data/raw/*.csv` im Repo-Root vorhanden
- Zusätzlich: `pip install dash`

## Starten

```bash
# vom Repo-Root aus:
python presentation/app.py

# oder direkt aus dem Ordner:
cd presentation
python app.py
```

Dann im Browser öffnen: **http://127.0.0.1:8050**

> Beim ersten Start werden alle Modelle einmalig berechnet (~30–60 Sekunden).
> Danach reagiert die App auf Tab-Wechsel ohne Verzögerung.

## Struktur

```
presentation/
├── app.py              ← Einstiegspunkt (Dash-Layout + Callback)
├── data_loader.py      ← Datensätze laden & aufbereiten
├── plots/
│   ├── apple.py        ← Apple-Modell & Plots
│   ├── weather.py      ← Weather-Modell & Plots
│   ├── gold.py         ← Gold-Modell & Plots
│   └── var_model.py    ← VAR-Modell & Plots
└── layout/
    ├── components.py   ← Wiederverwendbare UI-Bausteine
    ├── tab_apple.py    ← Apple-Tab-Layout
    ├── tab_weather.py  ← Weather-Tab-Layout
    ├── tab_gold.py     ← Gold-Tab-Layout
    └── tab_var.py      ← VAR-Tab-Layout
```

## Tabs

| Tab | Inhalt |
|-----|--------|
| Apple Stock | Close-Preis · Log-Returns · ACF/PACF · ARIMA(2,0,2) · Residuen · Testset · Forecast |
| Wetterdaten | Zeitreihe · EDA · ACF/PACF · ARIMA(3,0,0)+Fourier · Residuen · Testset · Forecast |
| Goldpreis | Zeitreihe · Stationarität · Log-Returns · ARIMA(2,1,2) · Residuen · Forecast |
| VAR-Analyse | Gemeinsamer Datensatz · Korrelation · IRF · FEVD · Residuen · Forecast |
