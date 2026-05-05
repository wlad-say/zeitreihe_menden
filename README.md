# Zeitreihenanalyse – Vertiefung Business Analytics

> Universitätsprojekt im Rahmen der Lehrveranstaltung **Vertiefung Business Analytics**  
> Technische Hochschule Würzburg-Schweinfurt | Prof. Dr. Christian Menden | SoSe 2026

---

## Projektbeschreibung

Dieses Repository enthält die Ergebnisse einer uni- und multivariaten Zeitreihenanalyse. Jede Person analysiert eine eigene Zeitreihe mithilfe der **Box-Jenkins-Methode (ARIMA)** und erstellt eine geeignete Prognose. Abschließend wird ein automatisiertes Modellselektionsverfahren für alle Zeitreihen gemeinsam entwickelt.

### Teilaufgaben

| Teil | Beschreibung |
|------|-------------|
| Teil 1 | Professionelles Git Repository aufsetzen |
| Teil 2 | Univariate Zeitreihenanalyse (ARIMA) pro Person |
| Teil 3 | Multivariate Zeitreihenanalyse mit automatisierter Modellselektion |

---

## Gruppe
 
| Name | Aufgabe | Zeitreihe | Branch |
|------|---------|-----------|--------|
| Aida Halimi          | Deskriptive Analyse | Amazon Sales (Umsatzdaten) | `feature/person1-amazon`  |
| Wladislaw Saydullaev | Datenvisualisierung | Weather (Wetterdaten)      | `feature/person2-weather` |
| Mehmet Bekler        | Datenaufbereitung   | Apple Stock (Aktienkurse)  | `feature/person3-apple`   |
 
---
## Projektstruktur

```
├── data/
│   ├── raw/            # Originale Rohdaten (nicht im Repo enthalten)
│   └── processed/      # Bereinigte / transformierte Daten
│
├── results/
│   ├── plots/          # Grafiken (ACF, PACF, Forecasts)
│   └── metrics/        # Evaluationsmetriken
│
├── ts_menden_team03.ipynb
├── .gitignore
├── README.md
├── CONTRIBUTING.md
└── requirements.txt
```

---

## Setup & Installation

### Voraussetzungen

- Python 3.10 oder höher
- VS Code (empfohlen)
- Git

### Installation

1. Repository klonen:
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```

2. Virtuelle Umgebung erstellen und aktivieren:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac / Windows
   python -m venv venv
   source venv/bin/activate
   ```

3. Dependencies installieren:
   ```bash
   pip install -r requirements.txt
   ```

4. Jupyter Kernel registrieren:
   ```bash
   python -m ipykernel install --user --name=venv
   ```

5. Notebooks in VS Code öffnen und Kernel `venv` auswählen.

---

## Dependencies

Die wichtigsten verwendeten Libraries:

| Library | Verwendungszweck |
|---------|-----------------|
| `pandas` | Datenverarbeitung |
| `numpy` | Numerische Berechnungen |
| `matplotlib` / `seaborn` | Visualisierung |
| `statsmodels` | ARIMA, ACF, PACF, Stationaritätstests |
| `pmdarima` | Automatische ARIMA-Modellselektion |
| `scikit-learn` | Evaluationsmetriken |
| `jupyter` | Notebooks |

Alle Dependencies mit Versionen: siehe [`requirements.txt`](requirements.txt)

---

## Datenquellen

Die Zeitreihen wurden selbstständig recherchiert und heruntergeladen. Die Rohdaten befinden sich **nicht** im Repository, da sie zu groß sind oder unter Lizenzbestimmungen stehen. Die genaue Quelle ist jeweils im Notebook der zuständigen Person dokumentiert.

| Person | Datenquelle   | Zeitraum                  | Frequenz               |
|--------|---------------|---------------------------|------------------------|
| AH     | Kaggle        | 2022-04-13 - 2023-06-29   | Irregularität          |
| WS     | Kaggle        | 01.01.2009 - 01.01.2017   | Zehn-Minuten Intervall |
| MB     | Yahoo Finance | 2016-01-04 - 2026-04-30   | Täglich                |

---

## Methodik

### Teil 2 – Univariate Analyse (ARIMA)
1. Stationaritätstest (ADF-Test, KPSS-Test)
2. Transformation der Zeitreihe (Differenzierung, Log-Transformation)
3. ACF & PACF Analyse
4. Modellselektion (Box-Jenkins-Methode + AIC/BIC)
5. Residualanalyse (Ljung-Box-Test)
6. Prognose für 10 Perioden inkl. Konfidenzintervalle

### Teil 3 – Multivariate Analyse
1. Automatisierter Loop über alle Zeitreihen und Modelle
2. Evaluationsmetriken: MAE, RMSE, MAPE
3. Auswahl des besten Modells
4. Gemeinsame Prognose für alle Zeitreihen

---

## Zeitplan & Abgabe

| Datum | Meilenstein |
|-------|------------|
| 04.05.2026 | Zwischenpräsentation 1 (5 Punkte) |
| 11.05.2026 | Zwischenpräsentation 2 (5 Punkte) |
| 18.05.2026 | Finalpräsentation (15 Punkte) + Abgabe via eLearning |

---

## Lizenz

Dieses Repository dient ausschließlich akademischen Zwecken im Rahmen der Lehrveranstaltung.  
© 2026 – Technische Hochschule Würzburg-Schweinfurt