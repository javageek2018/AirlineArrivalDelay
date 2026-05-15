## 🛫 Data Pipeline — From Raw Sources to Model-Ready Parquet

### 1. Acquire Flight Data from BTS

- **Source**: U.S. Bureau of Transportation Statistics (BTS) **On-Time Performance** database, accessed via the [TranStats portal](https://www.transtats.bts.gov/).
- **Coverage**: ~45M domestic commercial flight records spanning **2018–2024** (monthly CSV files).
- **Fields retained**: scheduled times, actual times, carrier, flight number, origin/destination airports, distance, and the `ArrDel15` delay flag.
- **Target definition**: `ArrDel15` = 1 when arrival is **≥ 15 minutes late**.

---

### 2. Acquire Weather Data from Iowa Mesonet ASOS

- **Source**: [Iowa Environmental Mesonet ASOS Download Service](https://mesonet.agron.iastate.edu/ASOS/).
- **Stations**: ~350 U.S. airports matched to BTS origin/destination codes.
- **Granularity**: hourly observations of temperature, dewpoint, humidity, wind speed/gust, visibility, precipitation, and weather codes.
- **Derived flags**: binary indicators for `is_rain`, `is_snow`, `is_fog`, `low_visibility`, `high_wind`, and `severe_weather`.

---

### 3. Join Weather to Flights (Leak-Safe)

- For each flight, attached weather observations at **scheduled departure time − 2 hours** for both origin and destination airports.
- The **2-hour lookback** ensures only **information knowable at prediction time** enters the feature set — no future data leakage.
- Created `*_weather_missing` flags for flights where no nearby observation was available.

---

### 4. Feature Engineering

- **Time features**: extracted `Year`, `Quarter`, `Month`, `DayOfWeek`, `dep_hour`, `arr_hour`, plus `_minus2` shifted versions for weather alignment.
- **Calendar flags**: `is_weekend`, `is_holiday`.
- **Historical delay rates** (computed strictly from past data to avoid leakage):
  - `carrier_delay_rate_30d` / `_90d`
  - `origin_delay_rate_30d` / `_90d`
  - `dest_delay_rate_30d` / `_90d`
- **Congestion features**: `origin_departures_3h` (flight density at origin).
- **Interaction features**: combined risk scores such as `wind_x_rain`, `combined_severe`, `holiday_x_hour`, `carrier_route_risk`.

---

### 5. Data Cleaning

- **Type casting**: standardized all numeric columns to `double` for `VectorAssembler` compatibility.
- **Null handling**: replaced sentinel/null values with sensible defaults (e.g., zero for missing precipitation, global mean for missing delay rates).
- **Deduplication**: removed duplicate flight records via composite keys (`FlightDate`, `Reporting_Airline`, `Flight_Number`, `Origin`, `Dest`).
- **Filter**: dropped rows with null target (`ArrDel15`) or missing scheduled times.
- **Leakage audit**: confirmed no post-departure features (e.g., actual `DepTime`) were included.

---

### 6. Temporal Train / Validation / Test Split

| Split | Years | Rows | Purpose |
|---|---|---|---|
| **Train** | 2018 – 2022 | ~31.1M | Model fitting |
| **Validation** | 2023 | ~6.7M | Early stopping, threshold tuning |
| **Test** | 2024 | ~6.9M | Final unbiased evaluation |

- **No row overlaps**: strict year-based partitioning to prevent temporal leakage.
- **Pre-COVID, COVID, and post-COVID** periods all represented in training for robustness.

---

### 7. Stratified Sampling for Training Efficiency

- Training data sampled at **~10% per (Year × Month × ArrDel15)** stratum to fit memory constraints while **preserving seasonality and class balance**.
- Validation similarly sampled (~15%) for fast early-stopping evaluation; **test set kept fully intact** for honest final metrics.

---

### 8. Write Model-Ready Parquet Files

- **Format**: Apache Parquet (columnar, compressed) — ~10× smaller and ~10× faster to read than CSV.
- **Output structure**:

      /data/processed/
        ├── train.parquet/   ← 2018-2022 stratified sample
        ├── val.parquet/     ← 2023 sampled
        └── test.parquet/    ← 2024 full

- **Partitioning**: by `Year` and `Month` for efficient filtering in downstream evaluation.
- **Final schema**: ~65 columns including target, raw features, derived weather flags, historical delay rates, and interaction features.

---

### ✅ Pipeline Properties

- **Reproducible**: deterministic via fixed random seeds.
- **Leakage-free**: 10 identified leakage risks audited and mitigated (temporal splits, historical rates from past data only, encoder isolation, no post-departure features).
- **Scalable**: handles 45M+ rows distributed via Spark.
- **Production-ready**: parquet outputs feed directly into the modeling pipeline (`VectorAssembler` → `StringIndexer` → `XGBoost`).
