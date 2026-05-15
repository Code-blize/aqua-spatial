# Aqua-Spatial Intelligence Platform

> Geospatial decision support for water infrastructure planning in Enugu State, Nigeria

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Google Earth Engine](https://img.shields.io/badge/GEE-Satellite%20Data-green)](https://earthengine.google.com)
[![Model](https://img.shields.io/badge/Model-RF%20%2B%20GB%20Ensemble-orange)](https://scikit-learn.org)
[![R²](https://img.shields.io/badge/R²-0.60-brightgreen)](https://github.com/Code-blize/aqua-spatial)

---

## The Problem

Water infrastructure decisions across Nigerian LGAs are routinely made without data. The result: boreholes drilled in the wrong locations, budgets wasted, and communities still walking kilometres for water. UNICEF estimates that over 60 million Nigerians lack access to basic water services - not because the resources don't exist, but because no one knows exactly where to deploy them.

**Aqua-Spatial solves the "where" problem.**

---

## What It Does

Aqua-Spatial is a spatial machine learning platform that tells Enugu State Government and UNICEF exactly which Local Government Areas need water infrastructure investment most urgently - and quantifies how confident we are in each prediction.

It combines:
- **Real satellite data** from 4 NASA/USGS sources via Google Earth Engine
- **WASH infrastructure data** structured after UNICEF field survey schemas
- **An ensemble ML model** trained on engineered geospatial features
- **Uncertainty quantification** so planners know where to trust the model and where to send field teams first

---

## Key Finding

> *A single satellite-derived feature - Heat-Vegetation Stress Index (LST ÷ NDVI from NASA Landsat 8 and MODIS) - accounts for **33.6% of all predictive power** in our model. More than any infrastructure metric.*

This means satellite data alone can identify water-stressed communities even before any ground surveys are conducted.

---

## Technical Architecture

```
NASA Landsat 8 (NDVI)          WASH Infrastructure Data
NASA SMAP (Soil Moisture)   +  (Synthetic, UNICEF-calibrated)
USGS SRTM (Elevation/Slope)
MODIS (Land Surface Temp)
         ↓                              ↓
    Google Earth Engine         GeoPandas Pipeline
    (17 LGA centroids)          (292 water points)
         ↓                              ↓
         └──────── Feature Engineering ─────────┘
                          ↓
              17 engineered features including:
              heat_veg_stress, drilling_difficulty,
              terrain_difficulty, soil_veg_index
                          ↓
         Random Forest + Gradient Boosting Ensemble
         Leave-One-Out Cross Validation (n=17)
         R² = 0.60 | MAE = 0.069
                          ↓
         Per-LGA Stress Score + Confidence Interval
         (3-source uncertainty quantification)
                          ↓
         Streamlit Decision Support Dashboard
```

---

## Satellite Features Extracted via GEE

| Feature | Source | Resolution | Why It Matters |
|---|---|---|---|
| NDVI | Landsat 8 (Dry season 2023) | 30m | Vegetation health = groundwater proxy |
| Soil Moisture | NASA SMAP SPL4SMGP | ~9km | Direct water availability signal |
| Elevation | USGS SRTM | 30m | Infrastructure feasibility |
| Slope | Derived from SRTM | 30m | Drilling and pipe-laying difficulty |
| Land Surface Temp | MODIS MOD11A1 | 1km | Heat stress = water demand |

---

## Model Performance

| Model | R² (LOOCV) | MAE |
|---|---|---|
| Random Forest | 0.5794 | 0.0687 |
| Gradient Boosting | 0.5720 | 0.0716 |
| **Ensemble (RF + GB)** | **0.6039** | **0.0687** |

Leave-One-Out Cross Validation was used as the evaluation strategy - the only statistically valid approach for a 17-sample geospatial dataset.

---

## Top 5 Most Water-Stressed LGAs

| LGA | Predicted Stress | Confidence | Key Signal |
|---|---|---|---|
| EnuguEast | 0.558 | Low | Lowest NDVI + highest LST |
| EnuguSouth | 0.547 | Low | Urban core heat stress |
| EnuguNorth | 0.499 | Low | Dense population, low functionality |
| Igbo-ezeSouth | 0.497 | Low | High terrain difficulty |
| **Udi** | **0.383** | **Medium** | 920km², steep slopes, low coverage |

**Udi is the most actionable intervention target** — high stress with medium confidence means the model is reliable enough to act on.

---

## Project Structure

```
aqua-spatial/
├── app/
│   ├── app.py                  # Streamlit dashboard
│   ├── dashboard_data.csv      # Final model outputs
│   ├── feature_importance.csv  # RF feature importance
│   ├── final_predictions.csv   # Full predictions + uncertainty
│   ├── rf_model.pkl            # Trained Random Forest
│   └── gb_model.pkl            # Trained Gradient Boosting
├── data/
│   └── enugu_data.pkl          # Enugu LGA geodata
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Data loading + EDA
│   ├── 02_gee_extraction.ipynb        # Google Earth Engine pipeline
│   ├── 03_feature_engineering.ipynb   # Feature construction
│   ├── 04_ml_model.ipynb              # Model training + evaluation
│   └── 05_uncertainty_analysis.ipynb  # Confidence intervals
├── requirements.txt
└── README.md
```

---

## Data Sources

| Layer | Source | Type |
|---|---|---|
| LGA Boundaries | GADM v4.1 | Real |
| Satellite Features | NASA/USGS via GEE | **Real** |
| Water Point Locations | Synthetic (UNICEF WASH schema) | Synthetic |
| Population Estimates | WorldPop Nigeria | Real |

> **Note:** Ground-truth water point data is synthetic, calibrated against UNICEF WASH Nigeria 2023 field survey benchmarks. The platform is designed to ingest real RUWASA field data with zero code changes.

---

## Installation

```bash
git clone https://github.com/Code-blize/aqua-spatial.git
cd aqua-spatial/app
pip install -r ../requirements.txt
streamlit run app.py
```

---

## Built With

- **Python** - GeoPandas, Rasterio, Scikit-learn, Pandas, NumPy
- **Google Earth Engine** - Satellite data extraction
- **Streamlit + Folium + Plotly** - Interactive dashboard
- **Scikit-learn** - Random Forest, Gradient Boosting, LOOCV

---

## Author

**Blessing Obasi-Uzoma** - Data Scientist  
GitHub: [@Code-blize](https://github.com/Code-blize)

---

## License

MIT License - free to use, adapt, and deploy for water security initiatives.
