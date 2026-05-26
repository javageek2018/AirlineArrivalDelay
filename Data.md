# Data Card: U.S. Flight Delay Dataset (2018–2024)

## Dataset Overview

- **Dataset name**: U.S. Flight Arrival Delay Dataset with Weather Features
- **Dataset version**: 1.0
- **Date created**: Ongoing tracking since 1987
- **Last updated**: December 2024
- **Curated by**: Group 5, DSC 288R Capstone
- **Repository**: https://github.com/javageek2018/AirlineArrivalDelay

### Summary

A combined dataset of ~45 million U.S. domestic flight records (2018–2024)
joined with airport-level hourly weather observations. Designed for binary
classification of flight delays (`ArrDel15` = arrival delayed ≥ 15 minutes)
with strict leakage prevention.

### Key Statistics

| Metric | Value |
|---|---|
| Total flight records | ~45M |
| Time span | 7 years (2018–2024) |
| Target variable | `ArrDel15` (binary) |
| Positive class rate | ~20.6% delayed |
| Number of features | 79 |
| Geographic coverage | ~350 U.S. commercial airports |
| Airlines covered | ~20 reporting carriers |

---

## Motivation

### Purpose

This dataset was created to train and evaluate machine learning models that
predict whether a scheduled flight will arrive ≥ 15 minutes late, using only
information knowable **before scheduled departure**. The dataset enables
operational, customer-facing, and analytical applications around proactive
disruption management.

### Specific Goals

- Provide a leakage-safe foundation for flight delay classification
- Include both flight metadata and environmental (weather) context
- Capture historical operational patterns (carrier, airport, route)
- Enable evaluation of model generalization across years (temporal holdout)

### Funding & Authorship

Bureau of Transportation Statistics & Iowa Mesonet ASOS

---

## Composition

### What does each row represent?

Each row represents a single scheduled commercial flight on a specific date,
including:

- Scheduled flight metadata (carrier, route, times, distance)
- Origin and destination airport weather conditions captured 2 hours before
  scheduled departure
- Trailing historical delay rates for the carrier, origin, and destination
- Derived interaction features and time-based encodings
- Binary target variable indicating whether arrival was ≥ 15 minutes late

### How many instances?

| Split | Years | Raw Rows | Sampled Rows |
|---|---|---|---|
| Train | 2018 – 2022 | ~31.1M | ~3.1M (10% stratified) |
| Validation | 2023 | ~6.7M | ~1M (15% stratified) |
| Test | 2024 | ~6.9M | ~6.9M (unsampled) |

### Are there labels?

Yes. The target variable `ArrDel15` is a binary label:
- `1` = arrival delayed ≥ 15 minutes from scheduled time
- `0` = arrived on time or within 15 minutes of schedule

Labels are derived from BTS-reported actual arrival times.

### Class Distribution

- On-time (class 0): ~79.4%
- Delayed (class 1): ~20.6%
- Distribution preserved across train/val/test via stratified sampling

### Is any information missing?

- `*_weather_missing` flags indicate flights where no weather observation
  was available within the lookback window
- Missing historical delay rates filled with the global training-set mean
- No rows with missing target values (filtered during preprocessing)
- Some carrier-route combinations have insufficient history for stable
  trailing rates (handled via fallback to airport-level rates)

### Are there errors, noise, or redundancies?

- BTS data quality is generally high; occasional reporting errors corrected
  via duplicate detection
- Weather observations may have gaps during instrument outages
- Some features are deliberately correlated (e.g., `dep_hour` and
  `dep_hour_minus2`) — the model handles redundancy via tree-based feature
  selection

### Does the dataset contain personal information?

No. The dataset contains:

- ❌ No passenger information
- ❌ No personally identifiable data
- ✅ Flight numbers (operational identifiers, not personal)
- ✅ Airline carrier codes (organizational identifiers)
- ✅ Airport codes (location identifiers, public infrastructure)

### Identifying information

Flight numbers and tail numbers were used during data integrity checks
(deduplication) but **excluded from final feature set** to prevent overfitting
to specific aircraft or flight schedules.

---

## Collection Process

### Data Sources

#### Primary: BTS On-Time Performance Database

- **Source**: U.S. Bureau of Transportation Statistics (BTS)
- **URL**: https://www.transtats.bts.gov/
- **Format**: Monthly CSV files
- **Acquisition**: Direct download via TranStats portal
- **License**: U.S. Government public domain
- **Fields used**: scheduled times, actual arrival times, carrier, origin,
  destination, distance, flight numbers

#### Secondary: Iowa Mesonet ASOS Weather Data

- **Source**: Iowa Environmental Mesonet (Iowa State University)
- **URL**: https://mesonet.agron.iastate.edu/ASOS/
- **Format**: Hourly observations via ASOS network
- **Acquisition**: Bulk download by airport station code
- **License**: Public domain
- **Fields used**: temperature, dewpoint, humidity, wind speed/gust,
  visibility, precipitation, weather codes

### How was the data acquired?

1. **BTS**: Programmatically downloaded monthly performance files (2018–2024)
2. **Weather**: Bulk-fetched ASOS data for ~350 airport stations matched to
   BTS origin/destination codes
3. **Integration**: Joined weather to flights at scheduled_departure_time − 2hr
4. **Feature engineering**: Computed rolling delay rates, derived interaction
   features, encoded categoricals

### Time of collection

- Data acquired: April 14, 2026
- Data spans: January 2018 through December 2024
- Last refreshed: May 3, 2026

### What sampling strategy was used?

- **Train**: Stratified 20% sample per (Year × Month × ArrDel15) stratum
  - Preserves seasonal patterns and class balance
  - Reduces ~31M rows to ~6.2M for training efficiency
- **Validation**: Stratified 15% sample by class on 2023 data
- **Test**: Full 2024 data retained (no sampling) for honest evaluation
- **Random seed**: 42 (deterministic for reproducibility)

### Were participants/stakeholders consulted?

Not applicable — public flight performance data with no individual
respondents.

### Any ethical review?

This dataset is derived entirely from publicly available government and
academic sources. No institutional ethical review was required.

---

## Preprocessing

### Steps Performed

1. **Source loading**
   - Loaded BTS CSV files into Spark DataFrames
   - Loaded ASOS weather data per airport

2. **Type casting**
   - All numeric features cast to `double` for `VectorAssembler` compatibility
   - Target `ArrDel15` cast to `int`

3. **Filtering**
   - Dropped rows with null `ArrDel15`
   - Dropped rows with null `CRSDepTime` (no scheduled departure time)
   - Removed cancellations and diverted flights

4. **Deduplication**
   - Composite key: `FlightDate + Reporting_Airline + Flight_Number_Reporting_Airline + Origin + Dest`
   - Removed exact duplicate rows

5. **Weather join**
   - Matched ASOS observations to flights at `scheduled_dep_time − 2hr`
   - Created `*_weather_missing` flags for unmatched flights
   - Filled missing numeric weather values with `0.0`

6. **Feature engineering**
   - Time features: `Month`, `DayOfWeek`, `dep_hour`, `arr_hour`, etc.
   - Calendar flags: `is_weekend`, `is_holiday`
   - Trailing delay rates: 30d and 90d windows for carrier, origin, dest
   - Congestion: `origin_departures_3h`
   - Interaction features: e.g., `wind_x_rain`, `combined_severe`

7. **Encoding**
   - StringIndexer applied to `Reporting_Airline`, `Origin`, `Dest`,
     `DepTimeBlk`
   - Indexers fit on training data only; applied to val/test

8. **Splitting**
   - Strict temporal split: 2018–2022 / 2023 / 2024
   - No overlap between splits

9. **Sampling**
   - Stratified random sampling on train and val (deterministic seed)

10. **Persistence**
    - Output written as Apache Parquet
    - Partitioned by `Year` and `Month` for downstream efficiency

### What was excluded?

- Cancelled flights (not relevant to arrival delay prediction)
- Diverted flights (arrival delay semantics differ)
- International flights (out of scope)
- Cargo and military flights (different operational patterns)
- Flights with corrupt or missing scheduled times

### What was NOT preprocessed?

Raw text fields like `origin_wx_codes` were retained in source data but not
used as features (redundant with derived boolean flags).

---

## Splits

### Strategy

**Time-based holdout** to simulate real-world deployment and detect temporal
drift.

| Split | Years | Purpose | Used For |
|---|---|---|---|
| Train | 2018–2022 | Learn patterns | Model fitting |
| Validation | 2023 | Tune & select | Early stopping, threshold selection, model comparison |
| Test | 2024 | Final eval | Single unbiased performance measurement |

### Rationale

- Temporal split simulates production deployment (predicting future from past)
- Includes pre-COVID, COVID, and recovery periods in training
- Validation captures "recent past" patterns (2023)
- Test on completely unseen 2024 data validates real-world readiness

### No overlap guaranteed

- Strict year-based partitioning
- Composite-key deduplication prevents row-level leakage
- StringIndexer fit on train only — never sees val/test

---

## Features

### Full feature list: 79 columns

#### Target

- `ArrDel15` (int): 1 if arrival delayed ≥ 15 min, else 0

#### Temporal (10)

- `Year`, `Quarter`, `Month`, `DayofMonth`, `DayOfWeek`
- `dep_hour`, `arr_hour`, `dep_hour_minus2`, `arr_hour_minus2`
- `is_weekend`, `is_holiday`

#### Flight Metadata (10+)

- `CRSDepTime`, `CRSArrTime`, `CRSElapsedTime`
- `Distance`, `DistanceGroup`
- `Reporting_Airline` (StringIndexed), `Origin` (StringIndexed),
  `Dest` (StringIndexed), `DepTimeBlk` (StringIndexed)

#### Weather — Origin (15)

- Continuous: `origin_temp_f`, `origin_dewpoint_f`, `origin_humidity`,
  `origin_feels_like_f`, `origin_wind_kts`, `origin_gust_kts`,
  `origin_visibility`, `origin_precip_in`
- Binary flags: `origin_is_rain`, `origin_is_snow`, `origin_is_fog`,
  `origin_low_visibility`, `origin_high_wind`, `origin_severe_weather`
- Quality: `origin_weather_missing`

#### Weather — Destination (15)

Same structure as origin, prefixed `dest_*`

#### Historical Delay Rates (9)

- `carrier_delay_rate_30d`, `carrier_delay_rate_90d`
- `origin_delay_rate_30d`, `origin_delay_rate_90d`
- `dest_delay_rate_30d`, `dest_delay_rate_90d`
- `origin_delay_rate`, `dest_delay_rate`, `origin_departures_3h`

#### Derived Interaction Features (10+)

- `hour_x_month`, `hour_x_dow`
- `wind_x_rain`, `lowvis_x_rain`, `combined_severe`
- `dep_per_hour_x_hour`
- `combined_route_risk`, `carrier_route_risk`
- `holiday_x_hour`, `is_longhaul`
- `route_delay_rate`, `route_n_flights`

### Feature value ranges

| Feature category | Range / type |
|---|---|
| Temperatures (°F) | typically −30 to 110 |
| Humidity (%) | 0–100 |
| Wind speed (kt) | 0–80+ |
| Visibility (mi) | 0–10 |
| Precipitation (in) | 0–5 |
| Delay rates | 0.0–1.0 |
| Binary flags | 0 or 1 |
| Hour features | 0–23 |
| Month | 1–12 |
| DayOfWeek | 1–7 |

---

## Intended Use

### Primary use cases

- Flight delay binary classification models
- Operational research and benchmarking
- Educational ML projects on imbalanced classification
- Time-series stability and generalization studies

### Suitable for

- ✅ Supervised binary classification (delay vs on-time)
- ✅ Imbalanced classification research (~20/80 split)
- ✅ Temporal generalization studies (train past → test future)
- ✅ Multi-source data integration case studies
- ✅ Feature importance and interpretability analysis

### NOT suitable for

- ❌ Real-time operational decisions requiring sub-hour latency
- ❌ International or non-U.S. flight prediction
- ❌ Safety-critical applications (recall ceiling ~80% with this data)
- ❌ Causal inference (correlational data only)
- ❌ Predictions for cancelled/diverted flights

---

## Distribution

### How is the data distributed?

- **Format**: Apache Parquet (compressed, columnar)
- **Storage**: [DBFS / S3 / local path]
- **Structure**:

      /data/processed/
        ├── train.parquet/   2018-2022 stratified sample
        ├── val.parquet/     2023 stratified sample
        └── test.parquet/    2024 full

- **Partitioning**: By `Year` and `Month` for efficient filtering
- **Compression**: Snappy (default Parquet)
- **Schema version**: 1.0

### How can users access it?

[Describe access — e.g., Databricks DBFS, S3 bucket URL, request process]

### License

- **BTS source data**: U.S. Government public domain
- **ASOS weather data**: Public domain (Iowa State University)

### Usage restrictions

None

---

## Maintenance

### Who maintains this dataset?

[Team name and contact]

### Update cadence

- **Annual**: Add new full year of flight data once available from BTS
- **As needed**: Update ASOS weather data if new stations come online
- **Quarterly**: Refresh historical delay-rate statistics if used in
  production scoring

### How will errors be communicated?

[Email list, GitHub issues, internal Slack channel]

### Versioning policy

- Major version (e.g., 2.0): Schema changes, new feature additions/removals
- Minor version (e.g., 1.1): New time periods added, bug fixes
- Patch version (e.g., 1.0.1): Documentation updates only

---

## Ethical Considerations

### Potential biases

- **Carrier representation**: Major carriers (e.g., AA, DL, UA) dominate the
  dataset; regional carriers have fewer rows
- **Geographic skew**: Major hubs (ATL, ORD, DFW, etc.) are overrepresented
- **Temporal coverage**: COVID year (2020) has reduced flight volume; may
  affect feature distributions

### Sensitive attributes

The dataset does not contain protected attributes (race, gender, religion,
etc.) since flights are not associated with individuals. However:

- Geographic patterns may correlate with socioeconomic factors
- Specific carriers may disproportionately serve certain communities
- Performance disparities by airline or airport tier have not been audited

### Recommended audits before deployment

- Per-carrier performance disparity analysis
- Per-airport-tier (large hub vs regional) analysis
- Per-region (urban vs rural) analysis
- Per-route-distance-class analysis

### Risks of misuse

- **Discriminatory pricing**: Using delay predictions for differential
  passenger pricing based on route patterns
- **Service refusal**: Refusing services to historically delayed routes
- **Operational discrimination**: De-prioritizing maintenance or staffing
  at consistently flagged airports

### Mitigations

- Document intended use cases clearly
- Add fairness monitoring to production deployments
- Restrict access for applications without clear ethical guardrails

---

## Known Issues & Caveats

### Data quality

- Sparse weather coverage at smaller regional airports
- Occasional gaps in ASOS observations during equipment outages
- BTS reporting lag of ~1 month for most recent data

### Methodological caveats

- Historical delay rates use **calendar-time rolling windows**, not
  flight-equivalent windows (low-volume carriers have noisier rates)
- 2-hour weather lookback is a design choice; longer lookbacks may
  improve realism but reduce signal
- Stratified sampling reduces training volume; full-data training may
  improve performance marginally

### Known limitations of the target

- `ArrDel15` is a binary threshold; doesn't capture severity of delay
- 15-minute threshold is industry-standard but somewhat arbitrary
- Cancellations not represented in this target (excluded entirely)

---

## Version History

| Version | Date | Changes |
|---|---|---|
| 1.0 | May 3, 2026 | Initial release covering 2018–2024 |

---

## References

- U.S. Bureau of Transportation Statistics On-Time Performance Database:
  https://www.transtats.bts.gov/
- Iowa Environmental Mesonet ASOS Network:
  https://mesonet.agron.iastate.edu/ASOS/
- Gebru, T., et al. (2018). *Datasheets for Datasets.*
- Pushkarna, M., Zaldivar, A., Kjartansson, O. (2022). *Data Cards: Purposeful
  and Transparent Dataset Documentation for Responsible AI.* FAccT '22.

---

## Acknowledgments

[Team members, advisors, data providers, funding sources]
