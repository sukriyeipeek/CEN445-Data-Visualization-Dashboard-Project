from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="NYC Yellow Taxi Dashboard", layout="wide")

DATA = Path(__file__).parent / "data"
TEMPLATE = "plotly_white"
TIP_THRESHOLD = 25

PAYMENT = {0: "Flex Fare", 1: "Credit card", 2: "Cash", 3: "No charge", 4: "Dispute"}
RATECODE = {1: "Standard", 2: "JFK", 3: "Newark", 4: "Nassau/Westchester", 5: "Negotiated", 6: "Group ride",
            99: "Unknown"}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PARTS = ["Night (0-5)", "Morning (6-10)", "Afternoon (11-15)", "Evening (16-20)", "Late evening (21-23)"]

def insight(text, scope="current filters"):
    st.caption(f"💡 **Insight ({scope}):** {text}")

@st.cache_data
def load_zones():
    return pd.read_csv(DATA / "zones.csv")

@st.cache_data
def load_data():
    raw = pd.read_parquet(DATA / "yellow_taxi_2025_01_sample.parquet")
    rep = {"Raw sample rows": len(raw), "Columns": raw.shape[1],
           "Missing cells (raw)": int(raw.isna().sum().sum())}
    df = raw.copy()
    df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    df["speed_mph"] = df["trip_distance"] / (df["duration_min"] / 60).replace(0, np.nan)

    rules = [
        ("Pickup outside January 2025", df["tpep_pickup_datetime"].between("2025-01-01", "2025-01-31 23:59:59")),
        ("Fare <= 0 or > $300 (refunds, errors)", df["fare_amount"].between(0.01, 300)),
        ("Distance <= 0 or > 100 miles", df["trip_distance"].between(0.01, 100)),
        ("Duration < 1 min or > 3 h", df["duration_min"].between(1, 180)),
        ("Speed > 80 mph", df["speed_mph"] <= 80),
        ("Total amount > $500 or negative", df["total_amount"].between(0, 500)),
    ]
    for name, keep in rules:
        rep[f"Removed: {name}"] = int((~keep.reindex(df.index, fill_value=False)).sum())
        df = df[keep.reindex(df.index, fill_value=False)]

    rep["Imputed passenger_count (missing or 0 -> 1)"] = int((df["passenger_count"].isna() | (df["passenger_count"] == 0)).sum())
    df["passenger_count"] = df["passenger_count"].fillna(1).clip(lower=1).astype(int)
    rep["Imputed RatecodeID (missing -> 1 Standard)"] = int(df["RatecodeID"].isna().sum())
    df["RatecodeID"] = df["RatecodeID"].fillna(1).astype(int)
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].fillna("N")
    for c in ["congestion_surcharge", "Airport_fee"]:
        df[c] = df[c].fillna(0)

    df["payment_name"] = df["payment_type"].map(PAYMENT).fillna("Other")
    df["ratecode_name"] = df["RatecodeID"].map(RATECODE).fillna("Unknown")
    df["pickup"] = df["tpep_pickup_datetime"]
    df["date"] = df["pickup"].dt.date
    df["hour"] = df["pickup"].dt.hour
    df["weekday"] = pd.Categorical(df["pickup"].dt.day_name(), DAYS, ordered=True)
    df["time_of_day"] = pd.cut(df["hour"], [-1, 5, 10, 15, 20, 23], labels=PARTS).astype(str)
    df["is_airport"] = (df["RatecodeID"].isin([2, 3]) | (df["Airport_fee"] > 0)).astype(int)
    tip_ok = (df["payment_type"] == 1) & (df["RatecodeID"] != 99)
    rep["Tip % masked: card trips with RatecodeID 99 (tip not recorded)"] = int(((df["payment_type"] == 1) & ~tip_ok).sum())
    df["tip_pct"] = np.where(tip_ok, (df["tip_amount"] / df["fare_amount"] * 100).clip(0, 100), np.nan)
    df["high_tip"] = np.where(df["tip_pct"].notna(), (df["tip_pct"] >= TIP_THRESHOLD).astype(float), np.nan)

    zones = load_zones()
    for pre, col in [("pu", "PULocationID"), ("do", "DOLocationID")]:
        z = zones[["LocationID", "Zone", "Borough"]].rename(
            columns={"LocationID": col, "Zone": f"{pre}_zone", "Borough": f"{pre}_borough"})
        df = df.merge(z, on=col, how="left")
        df[f"{pre}_zone"] = df[f"{pre}_zone"].fillna("Unknown")
        df[f"{pre}_borough"] = df[f"{pre}_borough"].fillna("Unknown")
    rep["Clean rows"] = len(df)
    rep["Missing cells (clean, excl. tip_pct)"] = int(df.drop(columns=["tip_pct", "high_tip"]).isna().sum().sum())
    return df.reset_index(drop=True), rep

df_all, prep_report = load_data()
zones = load_zones()

st.sidebar.title("🎛️ Filters")
dmin, dmax = df_all["pickup"].min().date(), df_all["pickup"].max().date()
date_range = st.sidebar.slider("Pickup date range", dmin, dmax, (dmin, dmax), format="MMM DD")
boroughs_all = sorted(df_all["pu_borough"].unique())
boroughs = st.sidebar.multiselect("Pickup borough", boroughs_all, default=boroughs_all)
pays_all = sorted(df_all["payment_name"].unique())
pays = st.sidebar.multiselect("Payment type", pays_all, default=pays_all)
parts = st.sidebar.multiselect("Time of day", PARTS, default=PARTS)
metric_name = st.sidebar.radio("Metric shown in charts", ["Trips", "Revenue (USD)", "Distance (miles)"])
airport_only = st.sidebar.checkbox("Airport trips only", value=False)

df = df_all[df_all["date"].between(*date_range) & df_all["pu_borough"].isin(boroughs)
            & df_all["payment_name"].isin(pays) & df_all["time_of_day"].isin(parts)].copy()
if airport_only:
    df = df[df["is_airport"] == 1]
if df.empty:
    st.warning("No trips match the current filters.")
    st.stop()
df["value"] = {"Trips": 1, "Revenue (USD)": df["total_amount"], "Distance (miles)": df["trip_distance"]}[metric_name]

with st.sidebar.expander("🧹 Preprocessing report"):
    for k, v in prep_report.items():
        st.write(f"**{k}:** {v:,}")

st.title("NYC Yellow Taxi — Exploratory Dashboard")
st.caption("CEN445 Introduction to Data Visualization · NYC TLC Trip Record Data, January 2025 · "
           "analysed on a 150,000-trip random sample of 3.47 M trips")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Trips (sample)", f"{len(df):,}",
          help="Trips in the 150,000-trip random sample that match the filters (after cleaning). The full month has 3.47 M trips, "
               "so multiply by about 23 to estimate the whole city.")
k2.metric("Revenue (sample)", f"${df['total_amount'].sum() / 1e6:,.2f} M",
          help="Sum of total_amount (fare, tip, tolls, surcharges) of the sampled trips only - not the city-wide revenue.")
k3.metric("Avg. fare", f"${df['fare_amount'].mean():.2f}")
k4.metric("Avg. distance", f"{df['trip_distance'].mean():.1f} mi", f"{df['duration_min'].mean():.0f} min avg. duration",
          delta_color="off")
k5.metric("Avg. tip (card)", f"{df['tip_pct'].mean():.1f} %" if df["tip_pct"].notna().any() else "n/a")


def style(fig, h=470, legend=True):
    fig.update_layout(template=TEMPLATE, height=h, margin=dict(l=10, r=10, t=50, b=10),
                      title_font_size=16, showlegend=legend)
    return fig

mname = metric_name.split(" (")[0].lower()

def fmt(v):
    return {"Trips": f"{v:,.0f} trips", "Revenue (USD)": f"${v:,.0f}", "Distance (miles)": f"{v:,.0f} mi"}[metric_name]

tab1, tab2, tab3, tab4, tab5 = st.tabs(["⏱️ Demand over time", "🗺️ Geography & flows", "💵 Fares & tips",
                                        "🤖 Machine learning", "📋 Data"])

with tab1:
    st.info("V1 heat-map, V2 time series and V3 sunburst will be added here by Sukriye Ipek.")

with tab2:
    st.info("V4 map, V5 Sankey and V6 network will be added here by Kaan Gokcek.")

card = df[df["tip_pct"].notna()]
with tab3:
    zrank = df.groupby(["pu_borough", "pu_zone"])["value"].sum()
    keep = set(zrank.groupby(level=0, group_keys=False).nlargest(10).index)
    t = df.assign(zone_grp=[z if (b, z) in keep else "Other zones" for b, z in zip(df.pu_borough, df.pu_zone)])
    MIN_CARD, TIP_MAX, NA_CODE = 30, 40, -1
    g = (t.groupby(["pu_borough", "zone_grp"])
           .agg(value=("value", "sum"), trips=("value", "size"), n_card=("tip_pct", "count"), tip_sum=("tip_pct", "sum"))
           .reset_index())
    gb = g.groupby("pu_borough")[["value", "trips", "n_card", "tip_sum"]].sum().reset_index()
    ids, labels, parents, vals, trips_, tips = ["NYC"], ["NYC"], [""], [g["value"].sum()], [g["trips"].sum()], []

    def tip_of(n_card, tip_sum):
        return min(tip_sum / n_card, TIP_MAX) if n_card >= MIN_CARD else NA_CODE

    tips.append(tip_of(g["n_card"].sum(), g["tip_sum"].sum()))
    for r in gb.itertuples():
        ids.append(r.pu_borough); labels.append(r.pu_borough); parents.append("NYC")
        vals.append(r.value); trips_.append(r.trips); tips.append(tip_of(r.n_card, r.tip_sum))
    for r in g.itertuples():
        ids.append(f"{r.pu_borough}|{r.zone_grp}"); labels.append(r.zone_grp); parents.append(r.pu_borough)
        vals.append(r.value); trips_.append(r.trips); tips.append(tip_of(r.n_card, r.tip_sum))
    tip_txt = ["n/a (< %d card trips)" % MIN_CARD if x == NA_CODE else f"{x:.1f} %" for x in tips]
    cut = (0 - NA_CODE) / (TIP_MAX - NA_CODE)
    rdylgn = px.colors.sample_colorscale("RdYlGn", [i / 10 for i in range(11)])
    scale = [[0, "#c8ccd0"], [cut - 1e-6, "#c8ccd0"]] + [[cut + (1 - cut) * i / 10, c] for i, c in enumerate(rdylgn)]
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=vals, branchvalues="total",
        customdata=np.column_stack([trips_, tip_txt]),
        marker=dict(colors=tips, colorscale=scale, cmin=NA_CODE, cmax=TIP_MAX,
                    colorbar=dict(title="Avg tip %", tickvals=[0, 10, 20, 30, 40])),
        texttemplate="%{label}<br>%{percentRoot:.1%}",
        hovertemplate="<b>%{label}</b><br>" + metric_name + ": %{value:,.0f}<br>Trips in sample: %{customdata[0]}"
                      "<br>Avg tip (card trips): %{customdata[1]}<extra></extra>"))
    fig.update_layout(title=f"V7 · Treemap — {metric_name} by pickup borough and zone "
                            f"(colour = avg tip % of card trips, grey = not enough card trips)")
    st.plotly_chart(style(fig, 560, False), width="stretch")
    bv = df.groupby("pu_borough")["value"].sum()
    tb = card.groupby("pu_borough")["tip_pct"].agg(["mean", "size"])
    tb = tb[tb["size"] >= 30]
    txt = f"{bv.idxmax()} is the biggest block ({bv.max() / bv.sum():.0%} of {mname})"
    if len(tb) > 1:
        hi, lo = tb["mean"].idxmax(), tb["mean"].idxmin()
        txt += (f"; average card tip is highest for {hi} pickups ({tb['mean'].max():.1f} %) and lowest for {lo} "
                f"({tb['mean'].min():.1f} %)")
        if tb["mean"].max() - tb["mean"].min() > 10:
            txt += " — a gap this large may reflect payment-recording differences, an important data caveat"
    insight(txt + ".")

    if len(card) > 50:
        s = card[card["trip_distance"] <= 30]
        s = s.sample(min(2000, len(s)), random_state=2)
        fig = px.parallel_coordinates(
            s, color="tip_pct", color_continuous_scale="Turbo", range_color=(0, 40),
            dimensions=["trip_distance", "duration_min", "speed_mph", "fare_amount", "tip_amount", "passenger_count"],
            labels={"trip_distance": "Distance (mi)", "duration_min": "Duration (min)", "speed_mph": "Speed (mph)",
                    "fare_amount": "Fare ($)", "tip_amount": "Tip ($)", "passenger_count": "Passengers"},
            title="V8 · Parallel coordinates — credit-card trips (colour = tip %) · drag on an axis to brush")
        st.plotly_chart(style(fig, 470, False), width="stretch")
        sh = card.loc[card["trip_distance"] < 1, "tip_pct"]
        lg = card.loc[card["trip_distance"] > 10, "tip_pct"]
        if len(sh) >= 30 and len(lg) >= 30:
            insight(f"brush the Distance axis: short trips (< 1 mile) are tipped {sh.mean():.0f} % of the fare, "
                    f"long trips (> 10 miles) {lg.mean():.0f} %. Colour = tip %, so you can see which trips end high on the Tip axis.")
        else:
            insight("brush any axis to isolate a group of trips; colour = tip %.")
    else:
        st.info("Not enough credit-card trips under the current filters for the parallel-coordinates plot.")

    c1, c2 = st.columns(2)
    with c1:
        if len(card) > 50:
            fig = px.violin(card[card["tip_pct"] <= 60], x="time_of_day", y="tip_pct", color="time_of_day", box=True,
                            points=False, category_orders={"time_of_day": PARTS},
                            title="V9 · Violin — tip % by time of day (credit cards)",
                            labels={"tip_pct": "Tip % of fare", "time_of_day": ""})
            fig.update_xaxes(tickangle=-20)
            st.plotly_chart(style(fig, 470, False), width="stretch")
    with c2:
        d = df[(df["trip_distance"] <= 20) & (df["fare_amount"] <= 100)]
        fig = px.density_heatmap(d, x="trip_distance", y="fare_amount", nbinsx=40, nbinsy=40,
                                 color_continuous_scale="Viridis", title="Distance vs. fare (trip counts)",
                                 labels={"trip_distance": "Distance (miles)", "fare_amount": "Fare ($)"})
        st.plotly_chart(style(fig, 470, False), width="stretch")
    parts_txt = []
    if len(card) > 50:
        tp = card.groupby("time_of_day")["tip_pct"].mean()
        parts_txt.append(f"tips are highest in the {tp.idxmax()} ({tp.max():.1f} %) and lowest in the {tp.idxmin()} ({tp.min():.1f} %)")
    if len(d) > 10:
        parts_txt.append(f"fare and distance correlate at r = {d['trip_distance'].corr(d['fare_amount']):.2f}, "
                         f"and {(df['RatecodeID'] == 2).mean():.1%} of trips use the JFK flat-rate code (the vertical stripe in the heat-map)")
    if parts_txt:
        insight("; ".join(parts_txt) + ".")

with tab4:
    st.info("K-Means clustering and the Random Forest tip model will be added here by Sukriye Ipek.")

with tab5:
    show = df.drop(columns=["value", "pickup", "date"])
    st.subheader("Filtered data (first 2,000 rows shown)")
    st.dataframe(show.head(2000), width="stretch", height=420)
    st.download_button("⬇️ Download filtered CSV", show.to_csv(index=False), "filtered_trips.csv", "text/csv")
    st.subheader("Summary statistics")
    st.dataframe(df[["trip_distance", "duration_min", "speed_mph", "fare_amount", "tip_amount", "total_amount",
                     "passenger_count"]].describe().round(2), width="stretch")
