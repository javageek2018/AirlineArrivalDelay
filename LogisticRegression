# Model Card — Flight Delay Prediction: Logistic Regression

**Version:** 1.0  
**Date:** June 2026  
**Notebook:** `04_logistic_regression.ipynb`  
**Artifact repository:** https://github.com/javageek2018/AirlineArrivalDelay  
**Model type:** Binary Logistic Regression (PySpark MLlib)  
**Task:** Predict whether a US domestic flight will arrive 15 or more minutes late (`ArrDel15 = 1`), assessed at T–2 hours before scheduled departure.

---

## 1. Model Overview

This model is one of three classifiers used for US domestic flight delay prediction. It serves as the linear baseline against which the tree-based models are compared. The model is implemented in PySpark and trained on a stratified sample of historical BTS flight records augmented with METAR weather data and engineered carrier/airport performance features.

**Prediction horizon:** All input features are provably observable at two hours before scheduled departure — no post-departure information leaks into the feature set.

**Pipeline:** `VectorAssembler` → `StandardScaler` → `LogisticRegression`

---

## 2. Intended Use and Scope

**Intended uses:**
- Academic benchmarking of a linear model against ensemble methods for flight delay classification.
- Establishing a performance floor for AUC-PR and monthly F1 stability that tree-based models must exceed to justify their added complexity.

**In-scope flights:** US domestic commercial flights with available T–2 weather observations at both origin and destination airports.

**Out-of-scope / known limitations:**
- **Not a real-time system.** The model requires weather observations at T–2; it cannot be used without them.
- **No carrier or route identity features.** Carrier, origin, and destination are encoded only through rolling aggregate rates, not as categorical identities. Performance on routes or carriers not well-represented in the training period may degrade.
- **Linear decision boundary.** The model cannot capture interaction effects (e.g., snow at origin *combined with* high congestion) without explicit feature engineering. Tree-based models handle these automatically.
- **US domestic only.** The training data covers US BTS records; generalisation to international or regional carriers is not validated.
- **Temporal scope:** Trained on pre-2023 data; validated on 2023; tested on 2024. Performance on years beyond 2024, or under large structural changes to the air travel system (e.g., network restructuring, new carriers), is unknown.

---

## 3. Training Data

| Property | Detail |
|---|---|
| **Source** | BTS On-Time Performance data + METAR weather observations, preprocessed in `FlightDelay_DataPrep.ipynb` |
| **Full training set** | 31,149,502 flights |
| **Training sample used** | 6,231,167 flights (stratified 20% sample, seed=42) |
| **Sampling method** | `sampleBy` stratified by `(Year × Month)` — 60 strata, 20% drawn from each to preserve seasonal and yearly distribution |
| **Training period** | Pre-2023 (years implied by the Year×Month strata structure) |
| **Class balance (sample)** | 5,116,995 not delayed (82.12%) / 1,114,172 delayed (17.88%) — imbalance ratio 4.59:1 |
| **Parquet file** | `flights_sample/train_sample.parquet` |

**Class imbalance handling:** A `class_weight` column is added at training time. Delayed flights (`ArrDel15=1`) are assigned a weight of 4.59 (the imbalance ratio); not-delayed flights receive weight 1.0. This is applied uniformly across all three models for a fair comparison.

**Null imputation:** Missing weather values (14 weather flag and continuous columns) are filled with 0 prior to pipeline fitting, representing a calm/clear baseline. The imputation is applied identically to training, validation, and test sets. No non-weather features had nulls.

---

## 4. Validation and Test Data

| Split | Period | Rows | Delay rate |
|---|---|---|---|
| **Validation** | 2023 | 6,743,403 | 20.56% |
| **Test** | 2024 | 6,965,246 | 20.82% |

- Validation is used for all tuning decisions (regularisation selection, threshold locking). It is not used for final reporting.
- The test set is the 2024 held-out set, touched exactly once after all modelling decisions are finalised. It represents a temporal out-of-distribution shift: the model was trained on pre-2023 data and must generalise to a different operational year.
- The slight increase in delay rate between val (20.56%) and test (20.82%) reflects mild year-on-year variation rather than a structural shift.

---

## 5. Features

The final model uses **24 features**, reduced from a broader initial set of 32 by dropping collinear and redundant columns.

| Group | Features | Count |
|---|---|---|
| Schedule / calendar | `Month`, `DayOfWeek`, `dep_hour`, `Distance`, `CRSElapsedTime`, `is_holiday` | 6 |
| Carrier rolling | `carrier_delay_rate_30d` | 1 |
| Airport congestion | `origin_departures_3h` | 1 |
| Target-encoded rates | `origin_delay_rate`, `dest_delay_rate` | 2 |
| Weather flags (origin) | `origin_is_rain`, `origin_is_snow`, `origin_is_fog`, `origin_severe_weather` | 4 |
| Weather flags (dest) | `dest_is_rain`, `dest_is_snow`, `dest_is_fog`, `dest_severe_weather` | 4 |
| Weather continuous (origin) | `origin_visibility`, `origin_wind_kts`, `origin_precip_in` | 3 |
| Weather continuous (dest) | `dest_visibility`, `dest_wind_kts`, `dest_precip_in` | 3 |

**Features explicitly dropped and rationale:**

| Dropped | Reason |
|---|---|
| `is_weekend` | Derivable from `DayOfWeek`; redundant |
| `carrier_delay_rate_90d` | Collinear with `carrier_delay_rate_30d` |
| `origin_low_visibility` | Captured by continuous `origin_visibility` |
| `origin_high_wind` | Captured by continuous `origin_wind_kts` |
| `dest_low_visibility` / `dest_high_wind` | Same rationale as origin flags |
| `origin_gust_kts` / `dest_gust_kts` | Collinear with `wind_kts` |

**Preprocessing:** All 24 features are assembled into a dense vector by `VectorAssembler` and then standardised to mean=0, std=1 by `StandardScaler`. The scaler is fit on training data only; the same fitted scaler is applied to validation and test without refitting. `standardization=False` is set on the `LogisticRegression` estimator to avoid double-scaling.

---

## 6. Model Architecture and Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Algorithm | `pyspark.ml.classification.LogisticRegression` | L-BFGS solver (default) |
| `regParam` | 0.0001 | Selected by sweep over [0.0001 … 0.5] |
| `elasticNetParam` | 0.0 | Pure L2 regularisation — no sparsity needed for a 24-feature dense set |
| `maxIter` | 100 | Model A converged in 27 iterations |
| `standardization` | False | Disabled to prevent double-scaling (pipeline scaler handles it) |
| `weightCol` | `class_weight` | 4.59× weight on delayed class |
| Decision threshold | **0.55** | Locked on 2023 val set by maximising class-1 F1 |

**Regularisation sweep:** Eight candidates were evaluated on the validation set across `regParam` values [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]. Both AUC-ROC and AUC-PR selected the same winner (Model A, `regParam=0.0001`). A warning was raised that the optimum is at the lower grid boundary; performance was essentially flat across the lower end of the grid, suggesting the model is not sensitive to further reduction.

**Full regularisation sweep results:**

| Model | regParam | AUC-ROC | AUC-PR | Iterations |
|---|---|---|---|---|
| A ✓ | 0.0001 | 0.6766 | 0.3533 | 27 |
| B | 0.0005 | 0.6766 | 0.3533 | 28 |
| C | 0.0010 | 0.6765 | 0.3533 | 27 |
| D | 0.0050 | 0.6764 | 0.3532 | 26 |
| E | 0.0100 | 0.6764 | 0.3532 | 24 |
| F | 0.0500 | 0.6760 | 0.3529 | 17 |
| G | 0.1000 | 0.6753 | 0.3522 | 14 |
| H | 0.5000 | 0.6704 | 0.3456 | 10 |

---

## 7. Threshold Tuning

The default threshold of 0.5 is calibrated for balanced classes. Given ~20% positive prevalence, lowering the threshold recovers delay recall at a precision cost. The threshold was swept across [0.20 … 0.85] and locked at the value maximising class-1 F1 on the 2023 validation set.

| Threshold | Class-1 F1 | Precision | Recall |
|---|---|---|---|
| 0.20 | 0.3420 | 0.2064 | 0.9986 |
| 0.35 | 0.3680 | 0.2293 | 0.9314 |
| 0.45 | 0.3975 | 0.2653 | 0.7929 |
| **0.55 ✓** | **0.4097** | **0.3166** | **0.5803** |
| 0.60 | 0.3961 | 0.3471 | 0.4613 |
| 0.70 | 0.3064 | 0.4196 | 0.2413 |

The locked threshold of **0.55** is applied unchanged to the 2024 test set. No re-tuning occurs after this point.

---

## 8. Evaluation Metrics and Results

The project uses a tiered metric hierarchy designed for class-imbalanced binary classification.

| Tier | Metric | Rationale |
|---|---|---|
| **1 — Primary** | AUC-PR | Imbalance-aware; random baseline = class prevalence (~0.20), not 0.5 |
| **1 — Primary** | ΔAUC-PR (val → test) | Temporal robustness signal |
| **2 — Selection** | AUC-ROC | Threshold-independent; used for regularisation selection |
| **2 — Stability** | Monthly class-1 F1 std dev | Within-year operational consistency |
| **3 — Operational** | Class-1 F1, Precision, Recall | At locked threshold; practical deployment metrics |
| **Reference only** | Accuracy, weighted F1 | Dominated by class-0 at 80% prevalence — not used for reporting |

### 8.1 Headline Results

| Metric | 2023 Validation | 2024 Test | Δ (val → test) |
|---|---|---|---|
| **AUC-PR** | **0.3533** | **0.3558** | **–0.0026** (improved) |
| AUC-ROC | 0.6766 | 0.6734 | +0.0032 |
| Class-1 F1 | 0.4097 | 0.4070 | +0.0026 |
| Class-1 Precision | 0.3166 | 0.3242 | –0.0076 |
| Class-1 Recall | 0.5803 | 0.5469 | +0.0334 |
| Monthly F1 std dev | 0.0693 | 0.0868 | +0.0175 |
| Random baseline (AUC-PR) | ~0.2056 | ~0.2082 | — |

**Key observation on temporal robustness:** ΔAUC-PR is −0.0026, meaning AUC-PR is marginally *higher* on the 2024 test set than on the 2023 validation set. This suggests the linear decision boundary learned from the training distribution generalises well to the mild year-on-year distributional shift between 2023 and 2024. However, monthly F1 standard deviation increases from 0.0693 to 0.0868 on test (+0.0175), indicating slightly less consistent month-to-month performance in 2024.

### 8.2 Confusion Matrix (2024 Test, threshold=0.55)

| | Predicted Not Delayed | Predicted Delayed |
|---|---|---|
| **Actual Not Delayed** | TN = 3,862,091 | FP = 1,653,189 |
| **Actual Delayed** | FN = 657,015 | TP = 792,951 |

At this operating point the model catches approximately **54.7% of delayed flights** (recall) while **32.4% of its delay predictions are correct** (precision). The false positive rate is notable — roughly 1 in 3 flagged flights is not actually delayed — a known trade-off when tuning for recall in an imbalanced setting.

### 8.3 Monthly Class-1 F1 — Validation (2023) vs Test (2024)

| Month | Val F1 | Test F1 | Val Prec | Val Recall | Test Prec | Test Recall |
|---|---|---|---|---|---|---|
| Jan | 0.3960 | 0.4148 | 0.2719 | 0.7285 | 0.3375 | 0.5381 |
| Feb | 0.3571 | 0.3275 | 0.3081 | 0.4246 | 0.2632 | 0.4333 |
| Mar | 0.4347 | 0.3937 | 0.3358 | 0.6163 | 0.3380 | 0.4714 |
| Apr | 0.4377 | 0.3798 | 0.3269 | 0.6619 | 0.2939 | 0.5365 |
| May | 0.3797 | 0.4675 | 0.2958 | 0.5302 | 0.3966 | 0.5692 |
| Jun | 0.4851 | 0.4575 | 0.4158 | 0.5821 | 0.3408 | 0.6957 |
| **Jul** | **0.5077** | **0.5068** | 0.3729 | 0.7952 | 0.3900 | 0.7234 |
| Aug | 0.4211 | 0.4356 | 0.2952 | 0.7341 | 0.2989 | 0.8031 |
| Sep | 0.3599 | 0.2941 | 0.3035 | 0.4422 | 0.2436 | 0.3713 |
| Oct | 0.3121 | 0.2054 | 0.2544 | 0.4035 | 0.2198 | 0.1928 |
| **Nov** | **0.2880** | 0.2907 | 0.2639 | 0.3170 | 0.2800 | 0.3022 |
| Dec | 0.3122 | 0.3763 | 0.2639 | 0.3822 | 0.3111 | 0.4760 |
| **Mean** | **0.3909** | **0.3791** | — | — | — | — |
| **Std dev** | **0.0693** | **0.0868** | — | — | — | — |

The model performs best in summer (July peak: F1=0.51 on both val and test), where high congestion, thunderstorm activity, and large delay volumes provide strong signal. It performs worst in the autumn/shoulder months (October–November), where lower delay rates and more diverse delay causes reduce the model's discriminative power.

---

## 9. Feature Importance and Explainability

Because logistic regression is a linear model with standardised features, the learned coefficients are directly interpretable: a larger absolute coefficient means the feature has a stronger influence on the log-odds of delay, and the sign indicates direction.

**Feature coefficients (sorted by absolute value):**

| Rank | Feature | Coefficient | Direction |
|---|---|---|---|
| 1 | `carrier_delay_rate_30d` | +0.4027 | Rolling 30-day carrier delay rate is the single strongest predictor — a carrier with a recent history of delays is more likely to delay again |
| 2 | `dep_hour` | +0.3210 | Flights later in the day face compounding delays from earlier rotations; afternoon/evening departures are higher risk |
| 3 | `dest_visibility` | −0.1577 | Better destination visibility reduces delay risk |
| 4 | `Distance` | +0.1479 | Longer flights have more exposure time to disruption |
| 5 | `origin_is_snow` | +0.1355 | Snow at origin significantly increases delay probability |
| 6 | `CRSElapsedTime` | −0.1064 | Scheduled block time — longer scheduled flights may have more buffer built in |
| 7 | `dest_wind_kts` | +0.0981 | Stronger headwinds at destination increase risk |
| 8 | `dest_is_rain` | +0.0830 | Rain at destination raises risk |
| 9 | `Month` | +0.0823 | Seasonal seasonality captured directly; positive coefficient reflects higher delay months |
| 10 | `dest_delay_rate` | +0.0741 | Destination airport's historical delay rate — congested hubs contribute risk |
| … | `is_holiday` | −0.0103 | Slight negative — possibly reflecting reduced traffic volume on holidays |

**Key interpretive notes:**
- Carrier performance (`carrier_delay_rate_30d`) dominates all weather signals combined, suggesting that operational reliability is more predictive than meteorological conditions in a linear model.
- The model's separation of destination and origin weather effects is appropriate: destination conditions (visibility, wind) affect landing and gate assignment; origin conditions (snow, rain) affect ground operations and pushback.
- The negative `CRSElapsedTime` coefficient seems counterintuitive but could be plausible. Longer scheduled segments often include more scheduled buffer time, which can absorb some upstream delays.
- `is_holiday` has a near-zero negative coefficient, which may reflect reduced congestion on actual holiday dates despite higher leisure travel before and after. This would likely be captured by `Month` and `DayOfWeek`).

---

## 10. Known Limitations and Failure Modes

**Where the model underperforms:**

- **Shoulder and winter months (Oct–Nov):** Monthly F1 drops to 0.21–0.29 on test. Delay patterns in these months are more heterogeneous and lower-volume, making them harder to predict with a linear boundary.
- **Rare extreme weather events:** The model treats weather as a set of binary flags and continuous scalars. Black-swan weather events (ice storms, severe hurricanes) that fall outside the training distribution will not be handled reliably.
- **Upstream knock-on delays:** The model captures carrier delay rates as rolling aggregates but does not model the specific inbound aircraft or crew rotation for any given flight. A flight whose inbound is severely delayed is not directly identifiable from these features.
- **New carriers or airports:** Routes and carriers not well-represented in the training strata will have unreliable target-encoded rates (`carrier_delay_rate_30d`, `origin_delay_rate`, `dest_delay_rate`), which are the model's most important signals.
- **Linear decision boundary:** The model cannot capture interaction effects between features (e.g., heavy snow *plus* high congestion is worse than either alone). This is the primary structural weakness relative to the Random Forest and XGBoost models.

**False positive rate:** At the operating threshold of 0.55, approximately 68% of flights the model flags as delayed are not actually delayed. For any operational use case where false alarms have a cost (e.g., preemptive passenger rebooking), this rate should be considered carefully.

---

## 11. Ethical Considerations

- The model does not use passenger-level data, demographic information, or any personally identifiable information. All features are flight-level operational and meteorological attributes.
- Target-encoded features (`carrier_delay_rate_30d`, `origin_delay_rate`, `dest_delay_rate`) reflect historical performance; carriers or airports serving underserved routes may have noisier estimates due to lower sample sizes.
- The model is intended for academic benchmarking. Any deployment in an operational or customer-facing setting would require additional validation, fairness analysis by route/carrier, and human oversight.

---

## 12. Model Versioning and Artifacts

| Artifact | Path | Description |
|---|---|---|
| Preprocessing pipeline | `Models/logreg/prep_pipeline` | Fitted `VectorAssembler` + `StandardScaler`; shared with RF and XGBoost notebooks |
| Model | `Models/logreg/lr_model` | Fitted `LogisticRegressionModel` (`regParam=0.0001`) |
| Metadata | `Models/logreg/metadata.json` | Hyperparameters, feature list, all val and test metrics, threshold |

**Runtime environment:**
- Python 3.10, PySpark 4.0.2, OpenJDK 17
- Google Colab (high-memory runtime), Spark driver memory 40 GB
- Training sample materialised via checkpoint to break Spark lineage

**Reproducibility:** The stratified training sample is deterministic (`seed=42`). Re-running the notebook on the same input parquet files will reproduce all metrics exactly.

---

## 13. Results Summary Card

```
Model         : Logistic Regression (PySpark MLlib)
Version       : 1.0
Target        : ArrDel15 (≥15 min late arrival)
Horizon       : T–2 hours before departure
Features      : 24
regParam      : 0.0001  |  elasticNetParam : 0.0
Threshold     : 0.55 (locked on 2023 val)

                      VAL (2023)     TEST (2024)      Δ
AUC-PR (primary)        0.3533         0.3558      –0.0026 
AUC-ROC                 0.6766         0.6734      +0.0032
Class-1 F1              0.4097         0.4070      +0.0026
Class-1 Precision       0.3166         0.3242
Class-1 Recall          0.5803         0.5469
Monthly F1 std          0.0693         0.0868      +0.0175
Random AUC-PR baseline  ~0.206         ~0.208
```
