# Aqua-Spatial Intelligence Platform
## Technical Methodology & Conceptual Guide

*A complete walkthrough of how the platform was built, why each decision was made, and what the results mean — written for both technical readers and curious minds.*

---

## Preface

This document explains the full methodology behind Aqua-Spatial — from the first line of data to the final confidence interval. It is written in two voices simultaneously: the technical explanation of *what* was done, and the plain-language explanation of *why it matters* and *what it means in the real world*.

By the end of this document, you should be able to explain every component of this project to a government official, a fellow data scientist, or your mother — and have all three understand it.

---

## Part 1 — The Problem We Are Solving

### 1.1 The Core Question

When a government or development organisation like UNICEF wants to build a new borehole, pipe system, or water kiosk in Enugu State — where do they put it?

Without data, the answer is: wherever is politically convenient, wherever is geographically easy, or wherever someone already requested one. This is how you end up with three boreholes in one neighbourhood and none in the community three kilometres away that has been walking 5km for water for a decade.

Aqua-Spatial answers a different question: **where does one naira of water infrastructure investment save the most lives?**

That is not a simple question. It requires understanding:
- Where are people currently underserved?
- Where is existing infrastructure failing?
- What does the terrain look like — is drilling even viable?
- How stressed is the environment in terms of heat and vegetation?
- And critically: how *confident* are we in any of these answers?

### 1.2 Why Enugu State Specifically

Enugu State presents an unusually complex water access challenge. It has 17 Local Government Areas (LGAs) ranging from dense urban cores (EnuguNorth, EnuguSouth — under 200km²) to vast rural territories (Udi at 920km², Uzo-Uwani at 899km²). The Udi Hills create genuine terrain barriers — steep slopes that make drilling difficult and pipe distribution expensive. And unlike flat-terrain states, water infrastructure decisions here cannot be made without accounting for geography.

This is exactly where data science adds value that a human planner alone cannot replicate.

---

## Part 2 — The Data

Every model is only as good as its data. Before writing a single line of machine learning code, we needed two types of information: *satellite observations* from space, and *ground-level water infrastructure records*.

### 2.1 Satellite Data — What We Pulled From Space

We used **Google Earth Engine (GEE)** — a cloud computing platform that gives researchers access to NASA and USGS satellite archives without downloading terabytes of imagery. Think of it as a search engine for satellite data.

We extracted four datasets:

---

**Dataset 1: NDVI from Landsat 8**
*What it is:* NDVI stands for Normalised Difference Vegetation Index. It measures how green and healthy vegetation is, derived from the near-infrared and red light bands of the Landsat 8 satellite.

*Why it matters for water:* Vegetation needs water to survive. Areas with consistently low NDVI in the dry season are areas where the land cannot support plant life — which is a strong signal of low groundwater availability and water stress.

*Technical detail:* We used the dry season (November 2023 – February 2024) to maximise the signal. During the rainy season, even water-scarce areas show green vegetation. The dry season reveals the underlying water availability of the land.

*Formula:*
```
NDVI = (Near-Infrared Band - Red Band) / (Near-Infrared Band + Red Band)
```
Values range from -1 to +1. Values above 0.3 indicate healthy vegetation. Values below 0.15 indicate bare soil, dry scrubland, or urban areas.

---

**Dataset 2: Soil Moisture from SMAP**
*What it is:* NASA's SMAP (Soil Moisture Active Passive) satellite directly measures the amount of water in the top 5cm of soil using microwave sensors.

*Why it matters:* This is as close to a direct water measurement as satellite remote sensing gets. Low surface soil moisture consistently means less water in the environment — less for vegetation, less for wells, less for communities.

*Technical detail:* SMAP has a native resolution of approximately 9km — coarser than Landsat's 30m. This is why we sampled at LGA centroid level rather than pixel level. At 9km resolution, LGA-level aggregation is the appropriate scale.

---

**Dataset 3: Elevation and Slope from SRTM**
*What it is:* The Shuttle Radar Topography Mission (SRTM) is a NASA dataset that mapped Earth's surface elevation in 2000 using radar from the Space Shuttle Endeavour. It remains the global standard for elevation data.

*Why it matters:* Two reasons. First, high elevation combined with steep slopes means water infrastructure is expensive and difficult — drilling rigs cannot access steep hillsides, and gravity-fed pipe systems require significant engineering. Second, Enugu's Udi Hills are a known terrain challenge that any serious infrastructure plan must account for.

*Derived feature — Slope:* Slope was computed from the elevation raster. A slope of 0 is flat ground. A slope of 1+ means steep terrain. Udenu has the highest slope in our dataset (0.94), followed closely by Udi (0.75).

---

**Dataset 4: Land Surface Temperature from MODIS**
*What it is:* MODIS (Moderate Resolution Imaging Spectroradiometer) is a NASA sensor aboard the Terra and Aqua satellites that measures the heat radiating from Earth's surface.

*Why it matters:* High land surface temperature (LST) in the dry season indicates heat-stressed, water-deficient land. Urban areas and bare soils absorb and re-emit more heat than vegetated, moisture-rich land. EnuguNorth and EnuguSouth consistently show the highest LST in our dataset (31°C) — the urban heat island effect combined with low vegetation.

---

### 2.2 Water Point Data — Ground Truth

The ideal ground-truth dataset would be RUWASA Enugu's official borehole registry — every water point in the state, its GPS coordinates, its construction date, its current functional status, and the population it serves.

This data was not accessible at the time of development. Attempts to access it through HDX (Humanitarian Data Exchange) and WPdx (Water Point Data Exchange) returned either 403 errors or no Enugu-specific records.

**What we did instead:** We generated a *synthetic* dataset calibrated against UNICEF WASH Nigeria 2023 field survey benchmarks. This means:
- Functionality rates (~52% functional) mirror real UNICEF-reported rates for comparable Nigerian states
- Water source type distribution (boreholes, hand pumps, shallow wells) reflects actual field survey proportions
- Depth-to-water variation by terrain follows known hydrogeological patterns for southeastern Nigeria

The synthetic data is clearly labelled throughout the platform. The architecture is designed so that replacing the synthetic CSV with real RUWASA data requires zero code changes.

---

### 2.3 The Bounding Box Problem

When generating water point coordinates, we initially used a *bounding box* — a rectangle around Enugu State — and placed points randomly inside it.

The problem: Enugu State is not a rectangle. It is an irregular polygon. Roughly 40% of the bounding box area falls *outside* Enugu's actual boundaries. Points that landed outside were orphaned — they had coordinates but no LGA assignment.

**Fix 1 — Nearest Neighbour Assignment:** Orphaned points were assigned to the nearest LGA centroid by geographic distance. This rescued approximately 132 points that would otherwise have been discarded.

**Fix 2 — Oji-River Manual Patch:** By random chance, zero synthetic points landed inside Oji-River's polygon. Rather than accept a blank LGA, we used a *polygon-constrained* generation method — generating candidate points and testing whether each one falls inside the polygon before accepting it. This produced 12 valid points for Oji-River.

These fixes are not cosmetic. A model trained on data with missing LGAs would produce unreliable predictions for those areas. Data quality is a prerequisite for model quality.

---

## Part 3 — Feature Engineering

Raw data is rarely what you train a model on. *Feature engineering* is the process of transforming raw measurements into variables that better capture the underlying patterns you want the model to learn.

This is where domain knowledge becomes data science.

### 3.1 What Is a Feature?

A feature is any variable used as input to the model. Raw features are things like "elevation = 310m" or "NDVI = 0.22". Engineered features are *combinations* of raw variables that capture something the raw variables alone cannot express.

### 3.2 The Features We Engineered

**heat_veg_stress = LST ÷ NDVI**

This is the single most important feature in the entire model (33.6% of variance).

The intuition: high temperature *and* low vegetation together is a much stronger signal of water stress than either alone. An area with high temperature but lush vegetation is not necessarily water-stressed — trees can survive heat if they have water. An area with low NDVI but moderate temperature might just be an urban area. But high temperature *combined with* low vegetation means the land is simultaneously hot and bare — the signature of a genuinely water-deficient environment.

By dividing LST by NDVI, we create a single number that captures both conditions simultaneously. EnuguNorth has the highest heat_veg_stress in the dataset (278.8), driven by its urban LST of 31°C and extremely low NDVI of 0.11.

**drilling_difficulty = avg_depth_m × slope**

This approximates the real-world cost of drilling a new borehole. Deeper wells cost more. Steeper terrain costs more. Multiplying them gives a compound difficulty index. Oji-River has the highest drilling difficulty (33.6) because it combines moderate depth with high slope — expensive terrain for infrastructure.

**soil_veg_index = soil_moisture × NDVI**

Where both soil moisture and vegetation are low simultaneously, water stress is compounded. This interaction captures co-occurring deficits that neither variable captures alone.

**terrain_difficulty**

A normalised composite of elevation and slope:
```
terrain_difficulty = (elevation / max_elevation × 0.5) + (slope / max_slope × 0.5)
```
This creates a single 0-1 index of how difficult the terrain is for infrastructure. Udenu scores highest (0.78), followed by Nsukka (0.78) and Udi (0.74).

**infrastructure_gap, functionality_rate, repair_burden**

These are derived from the water point data:
- *functionality_rate* = percentage of water points currently operational
- *infrastructure_gap* = percentage non-functional
- *repair_burden* = percentage needing repair

Together they describe the current state of water infrastructure maintenance across each LGA.

### 3.3 The Target Variable — composite_stress_score

Before we can train a model, we need something to train it *to predict*. This is called the target variable.

We built a weighted composite stress score from five dimensions:

```
stress_score = 0.30 × infrastructure_failure_rate
             + 0.20 × contamination_rate
             + 0.20 × heat_veg_stress (normalised)
             + 0.15 × terrain_difficulty
             + 0.15 × (1 / point_density_per_km²)
```

Each component is normalised to a 0-1 scale before weighting. The weights reflect a deliberate prioritisation: infrastructure failure (what is already broken) matters most, followed equally by water quality and satellite stress signal, with terrain and coverage density as secondary factors.

This composite score becomes the number every LGA is ranked by — and the number the ML model learns to predict.

---

## Part 4 — The Machine Learning Model

### 4.1 What Is a Decision Tree?

Before explaining Random Forest and Gradient Boosting, you need to understand their building block: the *decision tree*.

A decision tree asks a series of yes/no questions about the features to make a prediction. For example:

```
Is heat_veg_stress > 150?
    YES → Is functionality_rate < 40%?
              YES → High stress (score ≈ 0.55)
              NO  → Medium stress (score ≈ 0.38)
    NO  → Is ndvi > 0.25?
              YES → Low stress (score ≈ 0.20)
              NO  → Medium-low stress (score ≈ 0.28)
```

One tree is fragile. Change one data point and the entire tree structure can shift. This is called *high variance* — the model is too sensitive to the specific training data.

### 4.2 Random Forest — Wisdom of the Crowd

Random Forest builds 200 decision trees simultaneously, with two sources of randomness:
1. Each tree is trained on a *random subset* of the data points
2. Each tree considers only a *random subset* of the features at each split

Then it averages all 200 predictions.

The randomness is the point. By forcing each tree to be different, their errors are uncorrelated. When you average 200 uncorrelated predictions, the errors cancel out and the signal remains.

*Mathematically:*
```
RF_prediction = (1/200) × Σ Tree_b(x)  for b = 1 to 200
```

The result is a prediction that is far more stable than any single tree — and a model that tells you *which features it relied on most* (feature importance).

### 4.3 Gradient Boosting — Learning From Mistakes

Gradient Boosting takes a completely different approach. Instead of building trees in parallel and averaging them, it builds trees *sequentially*, where each new tree specifically learns from the *errors* of the previous one.

Round 1: Build a simple tree. It makes errors.
Round 2: Build a tree that predicts the *errors* from Round 1.
Round 3: Build a tree that predicts the *remaining errors* from Round 2.
...repeat 100 times.

The final prediction is the sum of all trees — each one correcting what the previous one got wrong.

This is analogous to a student who reviews their wrong answers after each exam and focuses specifically on what they got wrong. The result is a model that progressively reduces *systematic* errors — called *bias*.

*The key hyperparameter* is the learning rate (0.05 in our model). A slow learning rate means each tree makes small corrections — slower to train, but more robust against overfitting.

### 4.4 The Ensemble — Combining Both

Random Forest is good at reducing *variance* (wild, unstable predictions). Gradient Boosting is good at reducing *bias* (systematic, consistent errors). They fail in different ways.

When two models fail differently, averaging them tends to outperform either alone. Our ensemble:

```
final_prediction = (RF_prediction × 0.5) + (GB_prediction × 0.5)
```

Results:
| Model | R² | MAE |
|---|---|---|
| Random Forest | 0.5794 | 0.0687 |
| Gradient Boosting | 0.5720 | 0.0716 |
| **Ensemble** | **0.6039** | **0.0687** |

The ensemble achieved R²=0.60 — explaining 60% of water stress variance across all 17 LGAs.

### 4.5 Leave-One-Out Cross Validation (LOOCV)

We have 17 LGAs. A standard 80/20 train-test split would give us 13 training rows and 4 test rows. Four data points is not enough to evaluate a model reliably.

LOOCV is the statistically correct solution for small datasets. The logic:

```
Round 1:  Train on LGAs 2–17.  Predict LGA 1.
Round 2:  Train on LGAs 1, 3–17.  Predict LGA 2.
...
Round 17: Train on LGAs 1–16.  Predict LGA 17.
```

Every LGA is predicted exactly once, by a model that never saw it during training. After 17 rounds, you have 17 out-of-sample predictions — the most honest evaluation possible at this sample size.

The R²=0.60 and MAE=0.069 we report are LOOCV scores — not in-sample scores. This is an important methodological distinction. In-sample scores measure how well the model memorised the training data. LOOCV scores measure how well it *generalises* to unseen data.

---

## Part 5 — Uncertainty Quantification

This is the feature that makes Aqua-Spatial a responsible planning tool rather than just a ranking table.

### 5.1 Why Uncertainty Matters

Consider two LGAs with the same predicted stress score of 0.40:
- LGA A: predicted from 25 water points, both models agree closely, trees show low variance → High confidence
- LGA B: predicted from 1 water point, models disagree substantially, trees show high variance → Low confidence

A planner should treat these very differently. LGA A is a strong evidence-based signal. LGA B is an early warning that needs field verification before investment.

Without confidence intervals, both look identical on a ranking table.

### 5.2 The Three Sources

**Source 1: Model Disagreement**
```
uncertainty_model = |RF_prediction - GB_prediction|
```
When two well-calibrated models trained on the same data disagree strongly, it usually means the data is ambiguous or insufficient for that LGA. High disagreement = high uncertainty.

**Source 2: Data Reliability**
```
uncertainty_data = 1 / √(total_water_points)
```
This is the standard statistical formula for the standard error of a proportion. More data points → smaller uncertainty. An LGA with 25 water points has √25 = 5 times more reliable statistics than one with 1 point. EnuguEast's 100% stress rate from a single water point is statistically meaningless — this formula correctly assigns it high uncertainty.

**Source 3: Tree Variance**
```
uncertainty_trees = std(predictions from all 200 RF trees)
```
Each of the 200 Random Forest trees gives a slightly different prediction. A tight cluster of predictions (low standard deviation) means the model is confident. A wide spread means the model itself is uncertain about this LGA, regardless of whether the two ensemble models agree.

**Combined Score:**
```
uncertainty = (model_disagreement × 0.40) 
            + (data_reliability × 0.35) 
            + (tree_variance × 0.25)
```

The weights reflect a deliberate choice: model disagreement is the most direct signal of prediction uncertainty, data reliability is the most important for real-world decision-making, and tree variance adds a within-model signal.

### 5.3 Interpreting the Results

Thresholds were set at the 33rd and 66th percentiles of the uncertainty distribution — making the High/Medium/Low split data-driven rather than arbitrary.

| Confidence | Meaning | Recommended Action |
|---|---|---|
| **High** | Strong evidence, models agree, sufficient data | Act on this prediction |
| **Medium** | Moderate evidence, some uncertainty | Plan intervention, collect supplementary data |
| **Low** | Weak evidence, high uncertainty | Do not allocate budget; send field teams first |

**The key insight:** Low confidence does not mean the LGA is fine. EnuguEast has the highest stress score AND the lowest confidence. This means: something is seriously wrong there, but we don't have enough data to be sure exactly what. That is itself actionable — it tells you where to send a field survey team immediately.

---

## Part 6 — Feature Importance

### 6.1 How Random Forest Measures Importance

At each split in each tree, the model chooses which feature to split on. It selects the feature that most reduces the prediction error (measured by Mean Squared Error). Feature importance is computed as:

```
importance(feature) = Σ (reduction in MSE caused by splits on this feature)
                      across all splits in all 200 trees
                      normalised to sum to 1.0
```

### 6.2 What Our Model Found

| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| 1 | heat_veg_stress | 33.6% | NASA satellite composite — dominant signal |
| 2 | functionality_rate | 15.9% | Current infrastructure state |
| 3 | total_stress_rate | 12.8% | Combined failure burden |
| 4 | ndvi | 7.5% | Vegetation health independently |
| 5 | soil_veg_index | 6.2% | Combined soil-vegetation signal |

**The headline finding:** Three of the top five features are satellite-derived or satellite-engineered. Without the GEE extraction, the model would lose more than half its predictive power. This validates the entire satellite data pipeline.

**The second finding:** heat_veg_stress — an *engineered* feature we created by dividing LST by NDVI — is more than twice as important as any individual raw feature. This is what feature engineering is for: creating variables that capture patterns the raw data cannot express.

---

## Part 7 — The Spatial Dimension

### 7.1 Why Geography Matters

Standard machine learning treats every observation as independent. Spatial data violates this assumption — nearby things tend to be more similar than distant things. This is called *spatial autocorrelation*.

In our context: an LGA that borders a water-stressed area is more likely to share similar groundwater conditions, terrain, and infrastructure patterns than an LGA on the other side of the state.

We did not explicitly model spatial autocorrelation (this would require spatial regression or kriging methods). However, the satellite features — particularly NDVI, soil moisture, and LST — implicitly capture spatial patterns because they are derived from continuous raster surfaces. The elevation and slope features explicitly encode terrain continuity.

### 7.2 Spatial Interpolation — IDW

In the original project conception, we planned to use Inverse Distance Weighting (IDW) to interpolate groundwater depth across unsampled locations.

IDW formula:
```
Z_p = Σ(z_i / d_i^β) / Σ(1 / d_i^β)
```
Where:
- Z_p = estimated value at unsampled location p
- z_i = known value at sample point i
- d_i = distance from sample point i to p
- β = power parameter (typically 2)

The idea: points closer to p have more influence on the estimate than distant points. As β increases, nearby points dominate more strongly.

We pivoted from IDW to ML ensemble because the ML approach is more robust to non-stationarity (the relationship between features and stress may change across the state) and produces confidence intervals that IDW does not.

### 7.3 Coordinate Reference Systems

All spatial data was processed in **EPSG:4326** (WGS84) — the standard geographic coordinate system using latitude and longitude in decimal degrees.

For area calculations (LGA km²), we reprojected to **EPSG:32632** (UTM Zone 32N) — a projected coordinate system that measures distances in metres. Geographic coordinate systems distort areas when used for distance or area calculations because degrees of longitude narrow toward the poles. Projected coordinate systems eliminate this distortion for a specific region.

This distinction matters: Udi's area of 920.75km² was calculated in UTM, not in geographic degrees. The error from using geographic coordinates would be approximately 2-5% at Enugu's latitude — small but methodologically incorrect.

---

## Part 8 — Interpreting the Results

### 8.1 The Full LGA Ranking

Reading the results correctly requires holding three numbers simultaneously:

1. **Predicted stress score** — how stressed is this LGA?
2. **Confidence interval** — how certain are we?
3. **Confidence label** — should we act, plan, or investigate?

The most actionable combination is **high stress + medium/high confidence**. The most concerning combination is **high stress + low confidence** — not because you should ignore it, but because it demands immediate field investigation.

### 8.2 The Udi Finding in Detail

Udi is the project's headline finding for a specific reason. It is not the highest stress score — EnuguEast, EnuguSouth, and EnuguNorth all score higher. But Udi is the most *actionable* finding because:

- **Score: 0.383** — elevated, in the intervention zone
- **Confidence: Medium** — reliable enough to act on
- **Area: 920km²** — the largest LGA in the state
- **Point density: 0.023/km²** — the most underserved by area
- **Elevation: 310m, Slope: 0.75** — the Udi Hills, confirmed by real SRTM data
- **NDVI: 0.22, LST: 29.8°C** — stressed but not the most extreme

Every satellite layer tells a consistent story: Udi is a large, hilly, vegetation-stressed, sparsely-served LGA with enough evidence to justify priority investigation. The model knows this. The satellites confirm it.

EnuguEast scores higher (0.558) but with Low confidence from a single water point. Udi scores lower but with Medium confidence from 21 water points. For resource allocation purposes, Udi is the stronger signal.

### 8.3 What R²=0.60 Actually Means

R² measures the proportion of variance in the target variable that the model explains.

```
R² = 1 - (sum of squared prediction errors / sum of squared deviations from mean)
```

R²=0.60 means: if you knew nothing and predicted the mean stress score for every LGA, you would have 100% of the variance unexplained. Our model explains 60% of that variance — a 60% improvement over the naive baseline.

For a 17-sample dataset evaluated with LOOCV, R²=0.60 is a strong result. It is not 0.95 — but anyone claiming 0.95 on 17 samples with LOOCV is overfitting, not learning. The honest score is 0.60.

MAE=0.069 means: on average, our predictions are 6.9 percentage points off on a 0-1 scale. For a planning tool where the output drives field investigation decisions rather than automated budget allocation, this precision is sufficient.

---

## Part 9 — Limitations and Future Work

An honest methodology document acknowledges what the model cannot do.

### 9.1 Current Limitations

**Synthetic ground data:** The water point records are simulated. Real RUWASA data would dramatically improve model accuracy and narrow confidence intervals. The platform is designed for this upgrade.

**17 observations:** LGA-level analysis means 17 data points. This is the correct spatial unit for government planning, but it constrains the statistical power of the model. District-level analysis with ward-level data would allow far more powerful modelling.

**Static snapshot:** The satellite data is from 2023. Water stress conditions change seasonally and annually. A production system would update satellite features quarterly.

**Spatial autocorrelation:** We did not explicitly model the spatial relationships between neighbouring LGAs. A Geographically Weighted Regression or spatial lag model would capture these effects.

**No demand modelling:** We modelled water stress (supply-side) but not water demand. Population growth projections, economic activity, and agricultural water use would add a demand layer to the analysis.

### 9.2 The Path to Production

With real RUWASA data, this platform becomes a genuine government planning tool in three steps:

1. Replace `dashboard_data.csv` with real borehole registry exports
2. Re-run the feature engineering and model training notebooks
3. Redeploy the dashboard

The architecture supports this. The methodology is designed for it.

---

## Closing

Aqua-Spatial is a demonstration of what happens when geospatial data science is applied to a problem that matters. It is not perfect. The ground data is synthetic. The sample is small. The model explains 60%, not 100%, of variance.

But it is honest, rigorous, and deployable. It uses real satellite data from real NASA sensors over the real landscape of Enugu State. It produces uncertainty estimates so that planners know exactly where to trust the model and where to investigate further. And it is designed to improve the moment real field data becomes available.

The methodology is sound. The findings are defensible. The platform is ready.

---

*Built by Blessing Obasi-Uzoma — github.com/Code-blize*
*Aqua-Spatial Intelligence Platform — Water for All 2026*
