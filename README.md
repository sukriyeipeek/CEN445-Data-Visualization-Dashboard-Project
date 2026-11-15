# NYC Yellow Taxi — Exploratory Data Visualization Dashboard

**Course:** CEN445 Introduction to Data Visualization · **Tools:** Python, Streamlit, Plotly, scikit-learn, NetworkX

An interactive dashboard that explores New York City yellow-taxi trips in January 2025: when people ride, where trips
start and end, how much they pay and tip, and which kinds of trips exist.

## Dataset

| | |
|---|---|
| Main data | **NYC TLC Trip Record Data – Yellow Taxi, January 2025** (`yellow_tripdata_2025-01.parquet`) – 3,475,226 trips × 20 columns. Source: <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page> |
| Zone names | `taxi_zone_lookup.csv` from NYC TLC (265 zones, 6 boroughs) |
| Zone shapes | NYC Open Data "Taxi Zones" polygons, <https://data.cityofnewyork.us/resource/8meu-9t5y.geojson> |
| Used in the app | **150,000-trip random sample** (seed 445) stored in `data/yellow_taxi_2025_01_sample.parquet`, so the app starts fast and the repo stays small (~6 MB). Re-create it with `prepare_data.py` and the full file. |
| Types | datetimes, numerical (distance, fare, tip, tolls, total, passengers), categorical (payment type, rate code, vendor, pickup/dropoff zone), geographic (zone → polygon/coordinates) |

### Preprocessing (`load_data()` in `app.py`; counts are shown in the sidebar)
- Rows removed: 3 pickups outside January 2025, ~6,300 with fare ≤ 0 or > $300 (refunds / errors), ~3,300 with distance ≤ 0 or > 100 miles, 500 with duration < 1 min or > 3 h, 5 with speed > 80 mph
- Imputed: ~18,900 missing/zero `passenger_count` (→ 1), ~17,900 missing `RatecodeID` (→ Standard), fee columns (→ 0)
- Engineered features: duration, speed, hour, weekday, time-of-day, airport flag, tip % (credit-card trips only – tips on cash trips are not recorded; card trips with RatecodeID 99 are also excluded, because they are 100 % zero-tip in the data, i.e. the tip was never recorded — this was the cause of the "almost no tips" in Bronx/Brooklyn), borough/zone names joined from the zone tables
- Result: 139,977 clean rows, no missing values (except tip % for non-card trips by design)

## Visualizations (9 + 2 ML views)

| # | Chart | Type | Advanced |
|---|---|---|:-:|
| V1 | Weekday × hour heat-map (avg per day) | heat-map | ✅ |
| V2 | Hourly/daily time series by borough, range slider + selector, log-axis toggle | time series | |
| V3 | Borough → time of day → payment (colour = avg fare) | sunburst | ✅ |
| V4 | Pickups / dropoffs by taxi zone on the real NYC map | geographic choropleth | ✅ |
| V5 | Pickup borough → dropoff borough → payment type | Sankey | ✅ |
| V6 | Flows between the busiest zones at real coordinates; NetworkX weighted degree, betweenness and community detection (colour/size switches, metrics table) | network graph | ✅ |
| V7 | Borough → zone, colour = avg tip % | treemap | ✅ |
| V8 | Distance, duration, speed, fare, tip, passengers | parallel coordinates | ✅ |
| V9 | Tip % by time of day + distance-vs-fare density | violin + density heat-map | |
| ML1 | K-Means (k slider) in 3-D PCA space + elbow curve + cluster profile | clustering | |
| ML2 | Random Forest: will a card rider tip ≥ 25 %? importance, confusion matrix, what-if gauge | classification | |

**Interactivity:** hover tooltips, click-to-drill (treemap, sunburst), zoom/pan (map, 3-D), range slider and selector buttons,
brushing on parallel-coordinates axes, legend toggling, and these controls — sidebar date slider, pickup-borough,
payment and time-of-day multi-selects, metric radio (trips / revenue / distance), airport-only checkbox, time-resolution select
box, log-axis checkbox, pickups/dropoffs radio, two network sliders, K-Means k slider and the what-if inputs. All charts and the insight line under each one react to the sidebar filters; only the ML model metrics are fixed (see ML notes).

**Consistency:** one colour per borough and per payment type across all charts, shared `plotly_white` template,
titles/axes/legends everywhere and a one-line insight under each chart.

## ML notes
- **K-Means** on log-distance, log-duration, speed, passengers and cyclic hour; silhouette score and elbow curve guide *k*.
- **Random Forest** (balanced classes, 75/25 stratified split) on card-paid trips. Hold-out ROC-AUC ≈ 0.71, 3-fold CV ROC-AUC ≈ 0.71 ± 0.00, versus ≈ 0.70 for a logistic-regression baseline — the forest is only slightly better, so the tip is only partly predictable from trip features. Duration and distance are the main drivers. The "high tip" cut-off (25 % of the fare) is a modelling choice.
- **Treemap tip colour (V7):** tip % exists only for credit-card trips, so each block is coloured with the mean over its own card trips; blocks with fewer than 30 card trips are grey ("n/a") instead of being filled with an average.
- **Sidebar filters and ML:** both models are trained once on the full cleaned sample (K-Means: 30,000 trips, Random Forest: 80,000 trips), so their metrics do not change with the filters. Only the K-Means scatter and cluster profile show the trips that match the current filters. This is also stated inside the app on the ML tab.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional – rebuild the sample from the full data:
`python prepare_data.py yellow_tripdata_2025-01.parquet taxi_zone_lookup.csv taxi_zones.geojson 150000`

## Tests

```bash
pip install pytest
pytest -q tests
```
Smoke and data-quality tests (Streamlit `AppTest`): the app runs without exceptions with several filter combinations, every chart has its computed insight, the cleaned data meets the size requirement and the cleaning rules (fare, distance, duration, speed, passengers), and tip % exists only for card trips with a recorded tip.

## Limitations
- **One month, one sample:** only January 2025 and a 150,000-trip random sample (4 % of 3.47 M trips). KPIs such as revenue refer to the sample, not the whole city; no seasonal comparison is possible.
- **Tips:** recorded only for credit-card trips, and not recorded for rate code 99 trips (excluded). Cash tips are invisible in the data, so tip statistics describe card riders only.
- **Congestion pricing** started on 5 Jan 2025; the dashboard shows the daily trend but does not isolate its effect.
- **ML:** the tip model is only slightly better than a logistic-regression baseline (AUC 0.71 vs 0.70); the 25 % "high tip" threshold is a modelling choice.
- **No external data** (weather, events) is used.

## Team contributions

| Member | Contribution |
|---|---|
| **Member 1 — Şükriye İpek** |  ML (K-Means + Random Forest, what-if tool); V1 heat-map, V2 time series, V3 sunburst;  one-page report |
| **Member 2 — Kaan Gökçek** | Zone/geo data preparation; V4 choropleth map, V5 Sankey, V6 geographic network; layout & colour consistency |
| **Member 3 — Yusuf İzzeddin Fırat** | Data sampling & preprocessing, sidebar filters; V7 treemap, V8 parallel coordinates, V9 violin/density; README |

## Project structure
```
app.py            Streamlit application
prepare_data.py   Builds the sample + zone files from the official sources
data/             Sample parquet, zones.csv, taxi_zones.geojson
tests/            Smoke and data-quality tests (pytest + Streamlit AppTest)
.streamlit/       Theme settings (config.toml)
requirements.txt  README.md  REPORT.md
```
