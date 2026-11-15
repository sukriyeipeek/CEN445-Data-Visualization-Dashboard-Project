# CEN445 – Exploratory Data Visualization Dashboard: One-Page Report

**Team:** Şükriye İpek (2022555033), Kaan Gökçek (2022555026), Yusuf İzzeddin Fırat (2022555040)· **GitHub:** https://github.com/sukriyeipeek/CEN445-Data-Visualization-Dashboard-Project

## Dataset and context
We use the **NYC TLC Yellow Taxi Trip Records for January 2025**, published by the NYC Taxi & Limousine Commission. The full file has 3.47 million trips and 20 columns: pickup and dropoff times and zones, distance, fare, tip, tolls, total amount, passenger count, payment type and rate code. Zone names come from the TLC zone lookup table, and zone shapes come from NYC Open Data. To keep the dashboard fast, we work on a random sample of **150,000 trips** (fixed seed 445).

**Preprocessing.** We removed about 10,000 records that cannot be real trips: fares of $0 or less or above $300 (6,257), distances of 0 or above 100 miles (3,258), durations under 1 minute or over 3 hours (500), speeds above 80 mph (5) and pickups outside January (3). All missing values belong to "Flex Fare" trips. Deleting them would remove a whole payment type, so we kept these rows and filled the gaps (passenger count = 1, rate code = Standard, fees = 0). We also added new columns: duration, speed, hour, weekday, time of day, airport flag and tip %. After cleaning, **139,977 trips** remain. The app shows the count for every rule in the sidebar.

## Goals of the analysis
1. **When** is taxi demand highest? 2. **Where** do trips start and end, and how are zones connected? 3. **What** do riders pay and tip, and what types of trips exist?

## Visualization techniques (Streamlit, Plotly, scikit-learn, NetworkX)
The dashboard has nine visualizations, and seven of them are advanced types: **V1** heat-map of weekday × hour · **V2** time series by borough · **V3** sunburst (borough → time of day → payment) · **V4** choropleth map of the taxi zones · **V5** Sankey diagram (pickup borough → dropoff borough → payment) · **V6** network graph of zone-to-zone flows · **V7** treemap (borough → zone, colour = tip %) · **V8** parallel coordinates · **V9** violin plot with a density heat-map. There are also two machine learning views. **K-Means** groups trips into clusters and shows them in a 3-D PCA plot. A **Random Forest** predicts if a card rider will tip 25 % or more, and a what-if tool lets the user test their own trip.

**Interactivity and design.** Every chart has hover tooltips, and many support zoom, pan, click-to-drill or brushing. The sidebar has a date slider, three multi-selects, a metric selector and an airport checkbox. These filters update all charts. Each borough and each payment type has the same colour in every chart. Each chart also has a one-line insight below it, which is recalculated when the filters change.

## Key insights
- **Demand:** The busiest hours are 17:00–19:00, with the peak at 18:00. The quietest hours are 04:00–05:00. Thursday and Saturday are the busiest days, and Monday is the quietest.
- **Geography:** About 89 % of pickups are in Manhattan, and 85 % of trips both start and end there. JFK Airport is the 4th busiest pickup zone.
- **Fares:** The average fare is almost the same all day ($17–18), but it changes a lot by location: $14.6 for Manhattan pickups and $50.7 for Queens pickups, because of the airports. JFK trips show the flat $70 fare.
- **Payment and tips:** 74.5 % of trips are paid by credit card. Card riders tip about 26 % on average. Short trips (under 1 mile) get about 32 %, and long trips (over 10 miles) get about 20 %.
- **Data-quality finding:** Card trips with rate code 99 ("unknown") always have a $0 tip, because the tip was not recorded. This made Bronx and Brooklyn look like they never tip, so we excluded these trips from the tip statistics.
- **Network:** Our NetworkX analysis shows that Midtown Center is the most important connecting zone, and the Upper East Side zones have the most traffic.
- **Clustering (k = 4):** The four groups are long/airport trips, group rides (about 3.8 passengers), short midday trips and short evening trips.
- **Tip model:** The Random Forest reaches a ROC-AUC of 0.71, and a simple logistic regression reaches 0.70. This means tips are only partly predictable. Trip duration and distance are the most important features.

## Technical challenges
- **Size:** 3.47 million rows are too slow for an interactive app, so we used a random sample and cached the preprocessing and the ML models.
- **Dirty data:** The raw data has impossible values, such as negative fares and a 200,000-mile trip.
- **Tips:** Tips are recorded only for card payments, so all tip statistics use card trips only.
- **Geography:** We had to join the trip data with zone names and zone shapes, and calculate the centre of each zone for the network graph.

## Individual contributions
**Şükriye İpek:** V1–V3, machine learning (K-Means, Random Forest, what-if tool), report · **Kaan Gökçek:** Zone and map data preparation, V4–V6, layout and colour consistency · **Yusuf İzzeddin Fırat:** Data sampling and preprocessing, sidebar filters, V7–V9, README.
