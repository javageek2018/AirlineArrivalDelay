# Model Card: Flight Arrival Delay Classifier (Random Forest)

## Model Details

- **Model name**: Flight Arrival Delay Classifier
- **Model version**: 1.0
- **Model type**: Binary classification (ensemble of decision trees)
- **Algorithm**: Random Forest via `pyspark.ml.classification.RandomForestClassifier`
- **Developed by**: Group 5, DSC 288R Capstone
- **Model date**: May 2026
- **Repository**: https://github.com/javageek2018/AirlineArrivalDelay

### Architecture

- Ensemble of 100 independent decision trees
- Each tree trained on a random feature subset (1/3 of features per split)
- Majority-vote aggregation across trees for final prediction
- Class imbalance handled via per-sample weighting (`weightCol="weight"`, delayed class weight = 3.5)
- Trained on Apache Spark (`local[*]`, 35 GB driver memory)

### Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `numTrees` | 100 | Number of trees in the ensemble |
| `maxDepth` | 12 | Maximum depth per tree |
| `maxBins` | 64 | Max bins for continuous feature discretization |
| `featureSubsetStrategy` | `"onethird"` | Random feature subset per split (~16 of 47) |
| `minInstancesPerNode` | 5 | Minimum leaf size to prevent over-specific splits |
| `weightCol` | `"weight"` | Column assigning 3.5 to delayed class, 1.0 to on-time class |
| `seed` | 42 | Reproducibility seed |

---

## Intended Use

### Primary Use Case

Predict whether a scheduled U.S. domestic flight will arrive **15 or more
minutes late** based on flight schedule metadata, historical airline/airport
performance, and current weather conditions known at least 2 hours before
scheduled departure.

### Intended Users

- Airline operations teams (proactive disruption management)
- Customer-facing notification systems (delay alerts)
- Travel planning tools (delay-risk surfacing)
- Aviation analytics platforms

### Intended Use Scenarios

- Generating delay-likelihood scores for upcoming flights
- Triaging flights for ops-team review
- Driving customer notification systems with appropriate threshold tuning
- Risk-based pricing input for travel insurance products

### Out-of-Scope Uses

- **Real-time tactical decisions** (e.g., live ATC rerouting) — model is not
  designed for sub-hour predictions or live operational interventions
- **International flights** — trained only on U.S. domestic flights
- **Safety-critical decisions** — model recall at chosen threshold may be
  insufficient for safety applications
- **Predictions outside the 2-hour departure window** — weather features rely
  on observations from this window; predictions outside it would degrade
- **Causal attribution** — model identifies correlated patterns, not causes

---

## Training Data

### Sources

- **Flight data**: U.S. Bureau of Transportation Statistics (BTS) On-Time
  Performance database, accessed via the TranStats portal
- **Weather data**: Iowa Environmental Mesonet ASOS (Automated Surface
  Observing Systems) hourly observations

### Temporal Coverage

- **Training**: 2018 – 2022 (includes pre-COVID, COVID, and recovery)
- **Validation**: 2023
- **Test**: 2024

### Volume

| Split | Years | Approximate Rows | Sampled? |
|---|---|---|---|
| Train | 2018–2022 | ~31.1M raw → ~6.2M sampled | Yes (20% stratified by Year × Month × class) |
| Validation | 2023 | ~6.7M | No (full set used for evaluation) |
| Test | 2024 | ~6.9M | No (full set retained for evaluation) |

### Class Distribution

- Positive class (`ArrDel15 = 1`): ~17.9% of sampled training flights
- Negative class (`ArrDel15 = 0`): ~82.1% of sampled training flights
- Class ratio: 4.59:1; handled via sample weighting (`weightCol`, weight=3.5 for delayed class),
  intentionally below the true ratio to balance precision/recall

### Training Year Distribution (Sampled)

| Year | Rows |
|---|---|
| 2018 | 1,413,764 |
| 2019 | 1,452,111 |
| 2020 | 880,452 |
| 2021 | 1,176,594 |
| 2022 | 1,308,246 |

### Preprocessing

- Type casting to `double` for numeric features and label column
- Null handling: zero-fill for missing weather observations; global training mean for missing historical rates
- Composite-key deduplication
- **Target encoding** for high-cardinality categoricals (computed from past data only):
  - `Origin` (384 airports) → `origin_delay_rate` (expanding window on train, full 2018–2022 mean on val/test), `origin_delay_rate_30d`, `origin_delay_rate_90d`
  - `Dest` (384 airports) → `dest_delay_rate` (same logic), `dest_delay_rate_30d`, `dest_delay_rate_90d`
  - `Reporting_Airline` (19 carriers) → `carrier_delay_rate_30d`, `carrier_delay_rate_90d`
- Raw categorical columns (`Origin`, `Dest`, `Reporting_Airline`, `CRSDepTime`, `CRSArrTime`, `DepTimeBlk`) dropped after encoding; scheduled times extracted into `dep_hour` and `arr_hour`

---

## Features (47 total)

### Temporal & Calendar (7)
| Feature | Description |
|---|---|
| `Month` | Month of flight (1–12) |
| `DayofMonth` | Day of month (1–31) |
| `DayOfWeek` | Day of week (1–7, DOT convention) |
| `dep_hour` | Scheduled departure hour (0–23), extracted from `CRSDepTime` |
| `arr_hour` | Scheduled arrival hour (0–23), extracted from `CRSArrTime` |
| `is_weekend` | 1 if Saturday or Sunday, 0 otherwise |
| `is_holiday` | 1 if within ±1 day of a US federal holiday, 0 otherwise |

### Flight Metadata (3)
| Feature | Description |
|---|---|
| `CRSElapsedTime` | Scheduled flight duration in minutes |
| `Distance` | Flight distance in miles |
| `DistanceGroup` | BTS distance bin |

### Weather — Origin (14)
Observations at origin airport 2 hours before scheduled departure (`T-2`):

| Feature | Description |
|---|---|
| `origin_temp_f` | Temperature (°F) |
| `origin_dewpoint_f` | Dew point (°F) |
| `origin_humidity` | Relative humidity (%) |
| `origin_feels_like_f` | Feels-like temperature (°F) |
| `origin_wind_kts` | Wind speed (knots) |
| `origin_gust_kts` | Wind gust speed (knots; ~75% missing, zero-filled) |
| `origin_visibility` | Visibility (miles) |
| `origin_precip_in` | Precipitation (inches, 1-hour accumulation) |
| `origin_is_rain` | 1 if rain/drizzle/thunderstorm (RA/DZ/TS codes) |
| `origin_is_snow` | 1 if snow/sleet/ice (SN/SG/PL/IC codes) |
| `origin_is_fog` | 1 if fog/mist/haze (FG/BR/HZ codes) |
| `origin_low_visibility` | 1 if visibility < 3 miles |
| `origin_high_wind` | 1 if gust speed > 25 knots |
| `origin_severe_weather` | 1 if any of: rain, snow, low visibility, high wind |

### Weather — Destination (14)
Same structure as origin, observed 2 hours before scheduled arrival, prefixed `dest_*`.

### Target Encodings & Historical Delay Rates (9)
| Feature | Description |
|---|---|
| `origin_delay_rate` | Leakage-safe target encoding for `Origin`: expanding window mean on train; full 2018–2022 mean on val/test |
| `dest_delay_rate` | Same logic for `Dest` airport |
| `carrier_delay_rate_30d` | 30-day rolling delay rate for `Reporting_Airline` (target encoding) |
| `carrier_delay_rate_90d` | 90-day rolling delay rate for `Reporting_Airline` |
| `origin_delay_rate_30d` | 30-day rolling delay rate at origin airport |
| `origin_delay_rate_90d` | 90-day rolling delay rate at origin airport |
| `dest_delay_rate_30d` | 30-day rolling delay rate at destination airport |
| `dest_delay_rate_90d` | 90-day rolling delay rate at destination airport |
| `origin_departures_3h` | Number of departures from origin in the 3-hour window before this flight (congestion proxy) |

> **Note**: High-cardinality categoricals (`Origin`, `Dest`, `Reporting_Airline`)
> are represented via target-encoded delay rates rather than raw categorical values.
> Interaction effects are captured implicitly through tree splits rather than
> as explicit engineered features.

---

## Feature Importances (Top 20)

| Rank | Feature | Importance (Gini) |
|---|---|---|
| 1 | `dep_hour` | 0.1623 |
| 2 | `carrier_delay_rate_30d` | 0.1554 |
| 3 | `arr_hour` | 0.0899 |
| 4 | `origin_delay_rate_30d` | 0.0837 |
| 5 | `dest_delay_rate_30d` | 0.0646 |
| 6 | `origin_severe_weather` | 0.0628 |
| 7 | `dest_severe_weather` | 0.0427 |
| 8 | `carrier_delay_rate_90d` | 0.0384 |
| 9 | `origin_precip_in` | 0.0379 |
| 10 | `dest_precip_in` | 0.0253 |
| 11 | `origin_is_snow` | 0.0240 |
| 12 | `origin_delay_rate_90d` | 0.0170 |
| 13 | `origin_temp_f` | 0.0169 |
| 14 | `Month` | 0.0149 |
| 15 | `dest_delay_rate_90d` | 0.0135 |
| 16 | `origin_feels_like_f` | 0.0132 |
| 17 | `origin_dewpoint_f` | 0.0129 |
| 18 | `dest_delay_rate` | 0.0108 |
| 19 | `origin_departures_3h` | 0.0105 |
| 20 | `dest_is_rain` | 0.0097 |

Departure hour and 30-day carrier delay rate are the two dominant signals,
together accounting for ~32% of total feature importance.

---

## Evaluation

### Validation Set Metrics (2023, Threshold = 0.50)

| Metric | Value | Notes |
|---|---|---|
| ROC-AUC | 0.6910 | Threshold-independent ranking quality |
| PR-AUC | 0.3805 | ~2.13× lift over random baseline |
| Delayed Precision | 0.3377 | Of flagged delays, % actually delayed |
| Delayed Recall | 0.5584 | Of actual delays, % caught |
| On-Time Precision | 0.8624 | Of flagged on-time, % actually on-time |
| On-Time Recall | 0.7166 | Of actual on-time flights, % correctly identified |
| F1 (delayed class) | 0.4209 | Best F1 across threshold sweep |

### Test Set Metrics (2024, Threshold = 0.30)

| Metric | Value | Notes |
|---|---|---|
| ROC-AUC | 0.6878 | Threshold-independent ranking quality |
| PR-AUC | 0.3813 | ~1.83× lift over random baseline |
| Precision | 0.2361 | Of flagged delays, % actually delayed |
| Recall | 0.9278 | Of actual delays, % caught |
| F1 | 0.3764 | At threshold 0.30 |

### Test Set Confusion Matrix (2024, Threshold = 0.30)

|  | Predicted On-Time | Predicted Delayed |
|---|---|---|
| **Actual On-Time** | 1,161,661 | 4,353,619 |
| **Actual Delayed** | 104,707 | 1,345,259 |

- Actual delayed flights: 1,449,966
- Flagged as delayed: 5,698,878
- Delays caught (TP): 1,345,259 (92.8% of actual delays)
- False alarms (FP): 4,353,619 (76.4% of alerts sent)

### Operating Point

- **Default threshold**: 0.50 (best F1 on validation)
- **High-recall threshold**: 0.30 (used for test evaluation)
- **Selection rationale**: Threshold 0.30 prioritizes catching delays over
  precision — appropriate for customer alert systems where missing a delay
  is costlier than sending a false alarm
- **Flexibility**: Threshold can be re-tuned without retraining

### Threshold Sweep (Validation 2023)

| Threshold | Precision | Recall | F1 | Flagged |
|---|---|---|---|---|
| 0.50 | 0.3377 | 0.5584 | 0.4209 | 2,292,566 |
| 0.40 | 0.2770 | 0.7716 | 0.4077 | 3,862,060 |
| 0.35 | 0.2514 | 0.8648 | 0.3896 | 4,769,742 |
| 0.30 | 0.2313 | 0.9332 | 0.3707 | 5,594,659 |
| 0.25 | 0.2148 | 0.9809 | 0.3524 | 6,333,331 |
| 0.20 | 0.2061 | 0.9994 | 0.3417 | 6,724,797 |

### Generalization (Train → Validation)

| Metric | Train | Validation | Gap |
|---|---|---|---|
| ROC-AUC | 0.7112 | 0.6910 | −0.0202 |
| PR-AUC | 0.3798 | 0.3805 | +0.0007 |
| F1 | 0.7617 | 0.7083 | −0.0534 |
| Delayed Precision | 0.3498 | 0.3377 | −0.0121 |
| Delayed Recall | 0.4856 | 0.5584 | +0.0728 |

**Verdict**: Mild overfitting in F1 (gap of 0.05) due to tree depth of 12.
PR-AUC generalizes cleanly. No evidence of data leakage.

### Comparison vs Baselines

| Model | Test ROC-AUC | Test PR-AUC | Test F1 |
|---|---|---|---|
| Logistic Regression | 0.6766 | 0.3558 | 0.4071 |
| **Random Forest (this model)** | **0.6878** | **0.3813** | **0.3764** |
| XGBoost (selected) | 0.7099 | 0.4108 | 0.4393 |

> Note: Test F1 (0.3764) is evaluated at threshold 0.30, a high-recall
> operating point. At the F1-optimal threshold of 0.50, validation F1 is
> 0.4209.

---

## Limitations & Known Issues

### Performance Ceiling

- ROC-AUC of ~0.69 reflects the inherent difficulty of predicting delays
  with schedule + weather data alone. The model reaches a ceiling common
  to tree-based methods on this feature set.
- Improvements would require additional data sources (real-time radar,
  NOTAMs, aircraft maintenance signals).

### Precision-Recall Trade-Off

- At threshold 0.30, 76.4% of alerts are false positives. This precision
  level is unsuitable for high-cost automated actions.
- At the default 0.50 threshold, recall drops to 55.8% — many delays go
  undetected.

### Mild Overfitting

- Train F1 (0.7617) is noticeably higher than validation F1 (0.7083),
  suggesting depth-12 trees are learning some training-set-specific patterns.
  Reducing `maxDepth` or adding more trees may help.

### Seasonal Variation

- F1 varies across months due to shifting delay base rates. Performance is
  lower in fall/winter months.

### Geographic Coverage

- Trained only on U.S. domestic flights. Generalization to international
  routes is not validated.

### Temporal Validity

- Model trained on 2018–2022; validation 2023; test 2024. Annual retraining
  recommended to capture evolving carrier and weather patterns.

---

## Ethical Considerations

### Fairness

- Model has **not been audited** for performance disparities across airlines,
  regions, airport sizes, or socioeconomic factors.
- Recommended next step: per-airline and per-airport-tier performance audits
  before high-stakes deployment.

### Potential for Disparate Impact

- Historical delay rate features (`origin_delay_rate`, `dest_delay_rate`,
  `carrier_delay_rate_30d`) encode past performance patterns that may
  systematically disadvantage certain airlines or airports.
- Mitigation: monitor per-category false-positive rates.

### Privacy

- Training data contains no personally identifiable information.
- Flight numbers were used during feature engineering but not retained as
  model inputs.

### Misuse Risks

- Predictions could be misused for discriminatory pricing or service
  decisions (e.g., refusing service to historically delayed routes).
- Deployment should include guardrails against using model outputs for
  decisions that disadvantage specific groups of customers.

---

## Caveats & Recommendations

### Monitoring

- Track per-month F1, precision, and recall in production
- Alert on month-over-month F1 drops of >0.05
- Re-evaluate base-rate stability quarterly

### Retraining Cadence

- **Annually** at minimum to capture evolving carrier and weather patterns
- **More frequently** if monthly F1 stability degrades or significant
  operational changes occur (new carriers, route changes, etc.)

### Pre-Deployment Recommendations

1. Run fairness audit across airlines and airport categories
2. Compute calibration metrics (Brier score, reliability diagram)
3. Set up monitoring dashboards for ongoing performance tracking
4. Consider adding rolling target encodings at finer granularity (e.g.,
   route-level or hour × carrier) to close the gap with XGBoost
5. Tune `maxDepth` to reduce the train–validation F1 gap

### Threshold Recommendations by Use Case

| Use Case | Suggested Threshold | Recall | Precision |
|---|---|---|---|
| Maximum recall | 0.20 | ~99.9% | ~21% |
| High-recall alerts | 0.30 | ~93.3% | ~23% |
| Balanced default | 0.40 | ~77.2% | ~28% |
| Best F1 (default) | 0.50 | ~55.8% | ~34% |

---

## Pipeline & Reproducibility

### Data Pipeline

- Raw BTS data acquired from TranStats portal (monthly CSV files)
- Weather data acquired from Iowa Mesonet ASOS download service
- All processing in Apache Spark with deterministic random seeds (42)
- Output stored as Parquet (compressed, columnar)

### Training Pipeline

- Stratified sampling by (Year × Month × ArrDel15) at 20%
- Per-sample weighting (`scale_pos_weight = 3.5`) via `weightCol`
- VectorAssembler → RandomForestClassifier
- No early stopping — fixed at 100 trees

### Leakage Prevention

1. Strict temporal splits (no overlap between train/val/test years)
2. Historical delay rates computed using **past data only** (rolling backward windows)
3. `origin_delay_rate` / `dest_delay_rate` use expanding window on train; val/test use only the 2018–2022 training period mean
4. Weather observations lagged **2 hours before scheduled departure** (`T-2`)
5. No post-departure features (no actual times, taxi times, etc.)
6. No actual arrival time-derived columns
7. Composite-key deduplication
8. Stratified sampling preserves class distribution
9. Test set kept fully intact and never used for tuning

---

## Version History

| Version | Date | Notes |
|---|---|---|
| 1.0 | May 2026 | Initial production model |

---

