# Model Card: Flight Arrival Delay Classifier (XGBoost)

## Model Details

- **Model name**: Flight Arrival Delay Classifier
- **Model version**: 1.0
- **Model type**: Binary classification (gradient-boosted decision trees)
- **Algorithm**: XGBoost via `xgboost.spark.SparkXGBClassifier`
- **Developed by**: Group 5, DSC 288R Capstone
- **Model date**: May 2026
- **Repository**: https://github.com/javageek2018/AirlineArrivalDelay

### Architecture

- Gradient-boosted decision trees ensemble
- Up to 3,000 boosting rounds (early stopping selects optimal count)
- Maximum tree depth: 10
- Learning rate: 0.03
- Histogram-based tree construction (`tree_method='hist'`, `max_bin=256`)
- Trained on GPU (`device='cuda'`) via SparkXGBClassifier distributed
  training framework

### Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `n_estimators` | 3000 | Max trees; optimal count selected via early stopping |
| `learning_rate` | 0.03 | Conservative for careful convergence |
| `max_depth` | 10 | Captures deep feature interactions |
| `min_child_weight` | 3 | Prevents over-specific leaves |
| `subsample` | 0.8 | Row sampling per tree |
| `colsample_bytree` | 0.8 | Feature sampling per tree |
| `gamma` | 0.05 | Minimum split-gain threshold |
| `reg_lambda` | 1.0 | L2 regularization |
| `scale_pos_weight` | 1.0 | Imbalance handled via threshold tuning instead |
| `eval_metric` | `auc` | Optimized during training |
| `early_stopping_rounds` | 100 | Validation patience |

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
- **Safety-critical decisions** — model recall (~60% at chosen threshold) is
  insufficient for safety applications
- **Predictions ≥ 2 hours before scheduled departure** — weather features rely
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
| Train | 2018–2022 | ~31.1M raw → ~6.2M sampled | Yes (~20% stratified by Year × Month × class) |
| Validation | 2023 | ~6.7M raw → ~1M sampled | Yes (~15% stratified by class) |
| Test | 2024 | ~6.9M | No (full set retained for evaluation) |

### Class Distribution

- Positive class (`ArrDel15 = 1`): ~20.6% of flights
- Negative class (`ArrDel15 = 0`): ~79.4% of flights
- Class balance preserved during stratified sampling

### Preprocessing

- Type casting to `double` for numeric features
- Null handling: zero-fill for missing weather, global mean for missing
  historical rates
- Composite-key deduplication
- StringIndexer encoding for categorical features (Reporting_Airline,
  Origin, Dest, DepTimeBlk) — fit on training data only

---

## Features (79 total)

### Temporal & Calendar
`Year`, `Quarter`, `Month`, `DayofMonth`, `DayOfWeek`, `dep_hour`,
`arr_hour`, `dep_hour_minus2`, `arr_hour_minus2`, `is_weekend`, `is_holiday`

### Flight Metadata
`CRSDepTime`, `CRSArrTime`, `CRSElapsedTime`, `Distance`, `DistanceGroup`,
`Reporting_Airline` (indexed), `Origin` (indexed), `Dest` (indexed),
`DepTimeBlk` (indexed)

### Weather — Origin (15 features)
`origin_temp_f`, `origin_dewpoint_f`, `origin_humidity`,
`origin_feels_like_f`, `origin_wind_kts`, `origin_gust_kts`,
`origin_visibility`, `origin_precip_in`, `origin_is_rain`,
`origin_is_snow`, `origin_is_fog`, `origin_low_visibility`,
`origin_high_wind`, `origin_severe_weather`, `origin_weather_missing`

### Weather — Destination (15 features)
Same structure as origin, prefixed `dest_*`

### Historical Delay Rates
`carrier_delay_rate_30d`, `carrier_delay_rate_90d`,
`origin_delay_rate_30d`, `origin_delay_rate_90d`,
`dest_delay_rate_30d`, `dest_delay_rate_90d`,
`origin_delay_rate`, `dest_delay_rate`, `origin_departures_3h`

### Derived Interaction Features
`hour_x_month`, `hour_x_dow`, `wind_x_rain`, `lowvis_x_rain`,
`combined_severe`, `dep_per_hour_x_hour`, `combined_route_risk`,
`carrier_route_risk`, `holiday_x_hour`, `is_longhaul`,
`route_delay_rate`, `route_n_flights`

---

## Evaluation

### Test Set Metrics (2024, full set)

| Metric | Value | Notes |
|---|---|---|
| ROC-AUC | 0.7099 | Threshold-independent ranking quality |
| PR-AUC | 0.4108 | ~1.97× lift over random baseline |
| Accuracy | 0.6876 | At threshold 0.22 |
| Precision | 0.3501 | Of flagged delays, % actually delayed |
| Recall | 0.5739 | Of actual delays, % caught |
| Specificity | 0.7100 | Of on-time flights, % correctly identified |
| F1 | 0.4393 | F1-optimal balance |

### Operating Point

- **Threshold**: 0.22
- **Selection method**: F1-optimal on 2023 validation set
- **Rationale**: Balanced precision/recall trade-off chosen in the absence
  of explicit business cost asymmetry
- **Flexibility**: Threshold can be re-tuned without retraining for
  different use cases (e.g., 0.10 for high-recall, 0.50 for high-precision)

### Generalization (Validation → Test)

| Metric | Val (2023) | Test (2024) | Δ |
|---|---|---|---|
| ROC-AUC | 0.7133 | 0.7099 | −0.0034 |
| PR-AUC | 0.4102 | 0.4108 | +0.0006 |
| Delay base rate | 0.2062 | 0.2082 | +0.0020 |
| PR-AUC lift over baseline | 1.99× | 1.97× | −0.02× |

**Verdict**: Model generalizes cleanly to unseen 2024 data with no
meaningful degradation, and no hidden confounding from base-rate shifts.

### Monthly Stability (2024 Test)

| Statistic | F1 (Class 1) |
|---|---|
| Mean | 0.4137 |
| Std Dev | 0.0800 |
| Coefficient of Variation | ~19% |
| Min / Max | 0.21 (Oct) / 0.51 (Jul) |

**Interpretation**: Moderate seasonal variability. F1 tracks delay base
rate patterns rather than reflecting model fragility — underlying ROC-AUC
stays stable across months.

### Comparison vs Baselines

| Model | Test ROC-AUC | Test PR-AUC | Test F1 |
|---|---|---|---|
| Logistic Regression | 0.6766 | 0.3558 | 0.4071 |
| Random Forest | 0.6910 | 0.3813 | 0.3764 |
| **XGBoost (selected)** | **0.7099** | **0.4108** | **0.4393** |

---

## Limitations & Known Issues

### Performance Ceiling

- ROC-AUC of ~0.71 reflects the inherent difficulty of predicting delays
  with schedule + weather data alone. Industry benchmarks with the same
  feature space typically cap at 0.75–0.80.
- Improvements beyond this would require additional data sources
  (real-time radar, NOTAMs, aircraft-specific maintenance signals).

### Precision-Recall Trade-Off

- At the F1-optimal threshold, 65% of delay alerts are false positives.
  This precision level may be unsuitable for high-cost automated actions.
- For applications requiring higher precision, a higher threshold can be
  used, but recall drops correspondingly.

### Seasonal Variation

- Monthly F1 ranges from ~0.21 to ~0.51 across 12 months. Performance is
  weaker in fall/winter months despite the model handling weather features
  appropriately.

### Geographic Coverage

- Trained only on U.S. domestic flights. Generalization to international
  flights or non-FAA-jurisdiction airports is not validated.

### Temporal Validity

- Model trained on 2018–2022; validation 2023; test 2024. Performance on
  data beyond 2024 has not been verified. Annual retraining recommended.

### Probability Calibration

- Probabilities are reasonably calibrated due to `scale_pos_weight = 1`,
  but formal calibration metrics (e.g., Brier score, reliability diagrams)
  have not been computed.

---

## Ethical Considerations

### Fairness

- Model has **not been audited** for performance disparities across
  airlines, regions, airport sizes, or socioeconomic factors.
- Recommended next step: per-airline and per-airport-tier performance
  audits before high-stakes deployment.

### Potential for Disparate Impact

- Some categorical features (Origin, Dest, Reporting_Airline) could
  encode latent biases. Airlines or airports systematically flagged for
  higher delay risk may experience operational consequences.
- Mitigation: monitor per-category false-positive rates.

### Privacy

- Training data contains no personally identifiable information.
- Flight numbers were used during feature engineering but not retained
  as model inputs.

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
- Recompute monthly stability statistics annually

### Retraining Cadence

- **Annually** at minimum to capture evolving carrier and weather patterns
- **More frequently** if monthly F1 stability degrades or if significant
  operational changes occur (new carriers, route changes, etc.)

### Pre-Deployment Recommendations

1. Run fairness audit across airlines and airport categories
2. Compute calibration metrics (Brier, reliability diagram)
3. Set up monitoring dashboards for ongoing performance tracking
4. Define explicit business costs to revisit threshold selection

### Threshold Recommendations by Use Case

| Use Case | Suggested Threshold | Recall | Precision |
|---|---|---|---|
| High-recall alerts | 0.10 | ~80% | ~28% |
| Balanced default (current) | 0.22 | ~57% | ~35% |
| Operations monitoring | 0.30 | ~43% | ~41% |
| High-precision auto-actions | 0.50 | ~14% | ~56% |

---

## Pipeline & Reproducibility

### Data Pipeline

- Raw BTS data acquired from TranStats portal (monthly CSV files)
- Weather data acquired from Iowa Mesonet ASOS download service
- All processing in Apache Spark with deterministic random seeds (42)
- Output stored as Parquet (compressed, columnar) partitioned by Year/Month

### Training Pipeline

- Stratified sampling by (Year × Month × ArrDel15)
- StringIndexer + VectorAssembler + SparkXGBClassifier
- Early stopping on validation set (2023) via `validation_indicator_col`

### Leakage Prevention

10 leakage risks identified and mitigated:

1. Strict temporal splits (no overlap between train/val/test years)
2. Historical delay rates computed using **past data only**
3. Weather observations lagged **2 hours before scheduled departure**
4. StringIndexer fit on **training data only**
5. No post-departure features (no actual times, taxi times, etc.)
6. No actual arrival time-derived columns
7. Composite-key deduplication
8. Route statistics joined from training set only
9. Stratified sampling preserves class distribution
10. Test set kept fully intact and never used for tuning

---

## Version History

| Version | Date | Notes |
|---|---|---|
| 1.0 | May 2026 | Initial production model |

---

## References

- Mitchell, M., et al. (2019). *Model Cards for Model Reporting.* FAT* '19.
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.*
- U.S. Bureau of Transportation Statistics On-Time Performance database
- Iowa Environmental Mesonet ASOS network
