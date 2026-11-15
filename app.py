import json
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="NYC Yellow Taxi Dashboard", layout="wide")

DATA = Path(__file__).parent / "data"
TEMPLATE = "plotly_white"
TIP_THRESHOLD = 25

BOROUGH_COLORS = {"Manhattan": "#4C78A8", "Brooklyn": "#F2A93B", "Queens": "#54A24B", "Bronx": "#E45756",
                  "Staten Island": "#B279A2", "EWR": "#17BECF", "Unknown": "#9aa5b1", "N/A": "#9aa5b1"}
PAY_COLORS = {"Credit card": "#4C78A8", "Cash": "#54A24B", "Flex Fare": "#F2A93B",
              "Dispute": "#E45756", "No charge": "#9aa5b1"}

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

@st.cache_resource
def load_geojson():
    with open(DATA / "taxi_zones.geojson", encoding="utf-8") as f:
        return json.load(f)

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
geo = load_geojson()

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
    ndays = df.groupby("weekday", observed=True)["date"].nunique()
    piv = df.pivot_table(index="weekday", columns="hour", values="value", aggfunc="sum", observed=True)
    piv = piv.reindex(DAYS).fillna(0).div(ndays.reindex(DAYS), axis=0).round(1)
    fig = px.imshow(piv, aspect="auto", color_continuous_scale="YlOrRd", text_auto=False,
                    labels=dict(x="Pickup hour", y="", color=f"{metric_name}/day"),
                    title=f"V1 · Heat-map — average {metric_name} per day by weekday × hour")
    fig.update_xaxes(dtick=1)
    st.plotly_chart(style(fig, 400, False), width="stretch")
    hr = piv.sum(axis=0)
    wd = piv.sum(axis=1)[ndays.reindex(DAYS).notna()]
    insight(f"the busiest hour is {hr.idxmax()}:00 and the quietest is {hr.idxmin()}:00. "
            f"{wd.idxmax()} is the busiest weekday ({fmt(wd.max())} per day on average), "
            f"{wd.idxmin()} the quietest ({fmt(wd.min())}).")

    c1, c2 = st.columns(2)
    gran = c1.selectbox("Time resolution", ["Hourly", "Daily"], index=1)
    logy = c2.checkbox("Logarithmic y-axis", value=False)
    t = df.assign(t=df["pickup"].dt.floor("h" if gran == "Hourly" else "D"))
    ts = t.groupby(["t", "pu_borough"])["value"].sum().reset_index()
    fig = px.line(ts, x="t", y="value", color="pu_borough", color_discrete_map=BOROUGH_COLORS,
                  title=f"V2 · {gran} {metric_name} by pickup borough",
                  labels={"value": metric_name, "t": "", "pu_borough": "Borough"})
    fig.update_xaxes(rangeslider_visible=True,
                     rangeselector=dict(buttons=[dict(count=7, label="1w", step="day", stepmode="backward"),
                                                 dict(count=14, label="2w", step="day", stepmode="backward"),
                                                 dict(step="all")]))
    if logy:
        fig.update_yaxes(type="log")
    st.plotly_chart(style(fig, 520), width="stretch")
    tot_t = ts.groupby("t")["value"].sum()
    bsh = ts.groupby("pu_borough")["value"].sum()
    tf = "%b %d" if gran == "Daily" else "%b %d, %H:00"
    insight(f"{gran.lower()} {mname} range from {fmt(tot_t.min())} ({tot_t.idxmin():{tf}}) to "
            f"{fmt(tot_t.max())} ({tot_t.idxmax():{tf}}). {bsh.idxmax()} contributes {bsh.max() / bsh.sum():.0%} of the total"
            + ("; switch on the log axis to see the smaller boroughs. " if bsh.max() / bsh.sum() > .7 else ". ")
            + "Use the range slider or the 1w / 2w buttons to zoom.")

    sb = (df.groupby(["pu_borough", "time_of_day", "payment_name"])
            .agg(value=("value", "sum"), fare=("fare_amount", "mean")).reset_index())
    fig = px.sunburst(sb, path=["pu_borough", "time_of_day", "payment_name"], values="value", color="fare",
                      color_continuous_scale="Viridis", title=f"V3 · Sunburst — borough → time of day → payment "
                                                              f"({metric_name}, colour = avg fare $)")
    fig.update_coloraxes(colorbar_title="Avg fare $")
    st.plotly_chart(style(fig, 560, False), width="stretch")
    tod = df.groupby("time_of_day")["fare_amount"].mean()
    fb = df.groupby("pu_borough")["fare_amount"].agg(["mean", "size"])
    fb = fb[fb["size"] >= 30]
    txt = f"average fare varies between ${tod.min():.1f} and ${tod.max():.1f} across times of day"
    if len(fb) > 1:
        txt += (f", but between ${fb['mean'].min():.1f} ({fb['mean'].idxmin()}) and ${fb['mean'].max():.1f} "
                f"({fb['mean'].idxmax()}) across pickup boroughs")
    insight(txt + ". Click a ring to drill down.")

with tab2:
    which = st.radio("Map shows", ["Pickups", "Dropoffs"], horizontal=True)
    location_col = "PULocationID" if which == "Pickups" else "DOLocationID"
    zagg = df.groupby(location_col).agg(
        value=("value", "sum"), trips=("value", "size"), fare=("fare_amount", "mean")).reset_index()
    zagg = zagg.merge(zones[["LocationID", "Zone", "Borough"]],
                      left_on=location_col, right_on="LocationID", how="left")
    zagg[["Zone", "Borough"]] = zagg[["Zone", "Borough"]].fillna("Unknown")
    cap = float(zagg["value"].quantile(.95))
    fig = px.choropleth(
        zagg, geojson=geo, locations=location_col, color="value", range_color=(0, max(cap, 1e-9)),
        color_continuous_scale="YlOrRd", hover_name="Zone",
        hover_data={location_col: False, "Borough": True, "value": ":,.0f", "trips": ":,",
                    "fare": ":.2f"},
        labels={"value": metric_name, "trips": "Trips", "fare": "Avg fare $"},
        title=f"V4 · Geographic map — {which.lower()} by taxi zone "
              f"({metric_name}; colour capped at 95th percentile)")
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_traces(marker_line_width=0.3, marker_line_color="white")
    st.plotly_chart(style(fig, 640, False), width="stretch")
    t5 = zagg.nlargest(5, "value")
    share = t5["value"].sum() / zagg["value"].sum() if zagg["value"].sum() else 0.0
    insight(f"{t5.iloc[0]['Zone']} is the top {which.lower()[:-1]} zone "
            f"({fmt(t5.iloc[0]['value'])}); the top {len(t5)} zones by {mname} "
            f"account for {share:.0%} of the filtered total.")

    f1 = df.groupby(["pu_borough", "do_borough"])["value"].sum().reset_index()
    f2 = df.groupby(["do_borough", "payment_name"])["value"].sum().reset_index()
    nodes = {}

    def nid(kind, name):
        return nodes.setdefault((kind, name), len(nodes))

    src = [nid("pu", a) for a in f1.pu_borough] + [nid("do", a) for a in f2.do_borough]
    tgt = [nid("do", b) for b in f1.do_borough] + [nid("pay", b) for b in f2.payment_name]
    val = list(f1["value"]) + list(f2["value"])
    labels = [{"pu": f"{n} (pickup)", "do": f"{n} (dropoff)", "pay": n}[k] for k, n in nodes]
    ncol = [(PAY_COLORS if k == "pay" else BOROUGH_COLORS).get(n, "#9aa5b1") for k, n in nodes]
    lcol = [f"rgba({int(ncol[s][1:3], 16)},{int(ncol[s][3:5], 16)},"
            f"{int(ncol[s][5:7], 16)},0.4)" for s in src]
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, color=ncol, pad=14, thickness=16),
        link=dict(source=src, target=tgt, value=val, color=lcol)))
    fig.update_layout(title=f"V5 · Sankey — pickup borough → dropoff borough → payment type ({metric_name})")
    st.plotly_chart(style(fig, 520, False), width="stretch")
    flow_total = f1["value"].sum()
    same = f1.loc[f1["pu_borough"] == f1["do_borough"], "value"].sum() / flow_total if flow_total else 0.0
    big = f1.nlargest(1, "value").iloc[0]
    pay = f2.groupby("payment_name")["value"].sum()
    big_share = big["value"] / flow_total if flow_total else 0.0
    pay_share = pay.max() / pay.sum() if pay.sum() else 0.0
    insight(f"{same:.0%} of {mname} are within the same borough; the largest flow is "
            f"{big['pu_borough']} → {big['do_borough']} ({big_share:.0%}); "
            f"{pay.idxmax()} is the main payment type ({pay_share:.1%}).")

    st.markdown("#### V6 · Network — trip flows between the busiest taxi zones")
    n1, n2, n3, n4 = st.columns(4)
    topn = n1.slider("Number of busiest zones", 10, 60, 30, 5)
    minlink = n2.slider("Minimum trips on a link", 1, 40, 8)
    node_color = n3.radio("Colour nodes by", ["Borough", "Community"], horizontal=True)
    node_size = n4.radio("Size nodes by", ["Trips", "Betweenness"], horizontal=True)
    tot = pd.concat([df["PULocationID"], df["DOLocationID"]]).value_counts()
    top = set(tot.head(topn).index)
    fl = df[(df["PULocationID"] != df["DOLocationID"])
            & df["PULocationID"].isin(top) & df["DOLocationID"].isin(top)]
    a = np.minimum(fl["PULocationID"], fl["DOLocationID"])
    b = np.maximum(fl["PULocationID"], fl["DOLocationID"])
    ed = fl.groupby([a.rename("a"), b.rename("b")]).size().reset_index(name="n")
    located = zones.dropna(subset=["lat", "lon"]).set_index("LocationID")
    ed = ed[(ed["n"] >= minlink) & ed["a"].isin(located.index) & ed["b"].isin(located.index)]
    ed = ed.nlargest(250, "n")
    G = nx.Graph()
    for r in ed.itertuples():
        G.add_edge(int(r.a), int(r.b), weight=int(r.n), dist=1.0 / int(r.n))
    if G.number_of_edges():
        strength = dict(G.degree(weight="weight"))
        betw = nx.betweenness_centrality(G, weight="dist")
        comms = list(nx.community.greedy_modularity_communities(G, weight="weight"))
        comm_of = {n: i + 1 for i, c in enumerate(comms) for n in c}
        modularity = nx.community.modularity(G, comms, weight="weight")
    else:
        strength, betw, comm_of, comms, modularity = {}, {}, {}, [], 0.0

    fig = go.Figure()
    if len(ed):
        ed["bucket"] = np.ceil(5 * ed["n"] / ed["n"].max()).clip(1, 5).astype(int)
        for bk, g in ed.groupby("bucket"):
            xs, ys = [], []
            for r in g.itertuples():
                xs += [located.loc[r.a, "lon"], located.loc[r.b, "lon"], None]
                ys += [located.loc[r.a, "lat"], located.loc[r.b, "lat"], None]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                line=dict(width=0.8 + 1.7 * bk, color=f"rgba(90,100,110,{0.18 + 0.1 * bk})")))
    nd = located.loc[located.index.isin(G.nodes)].reset_index().copy()
    nd["trips"] = nd["LocationID"].map(tot)
    nd["strength"] = nd["LocationID"].map(strength).fillna(0)
    nd["betweenness"] = nd["LocationID"].map(betw).fillna(0.0)
    nd["community"] = nd["LocationID"].map(comm_of).fillna(0).astype(int)
    size_col = "trips" if node_size == "Trips" else "betweenness"
    max_size = nd[size_col].max() if len(nd) else 0.0
    nd["msize"] = 8 + 30 * nd[size_col] / max(max_size, 1e-9)
    if node_color == "Borough":
        groups = [(bor, BOROUGH_COLORS.get(bor, "#9aa5b1"), g) for bor, g in nd.groupby("Borough")]
    else:
        pal = px.colors.qualitative.Bold
        groups = [(f"Community {c}", pal[(c - 1) % len(pal)], g) for c, g in nd.groupby("community")]
    for name, colr, g in groups:
        fig.add_trace(go.Scatter(
            x=g["lon"], y=g["lat"], mode="markers", name=name,
            marker=dict(size=g["msize"], color=colr, line=dict(width=1.2, color="white")),
            text=g["Zone"],
            customdata=np.column_stack([g["trips"], g["strength"], g["betweenness"],
                                        g["community"], g["Borough"]]),
            hovertemplate="<b>%{text}</b> (%{customdata[4]})<br>%{customdata[0]:,} trip ends"
                          "<br>Weighted degree: %{customdata[1]:,}"
                          "<br>Betweenness: %{customdata[2]:.3f}"
                          "<br>Community: %{customdata[3]}"
                          f"<br>Graph modularity Q: {modularity:.3f}<extra></extra>"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1.32)
    fig.update_layout(
        title="Zones placed at their real coordinates · link width = trips in either direction · "
              f"node size = {node_size.lower()}, colour = {node_color.lower()}",
        plot_bgcolor="#f6f8fa")
    st.plotly_chart(style(fig, 640), width="stretch")
    if len(ed):
        strongest = ed.nlargest(1, "n").iloc[0]
        bridge = max(betw, key=betw.get)
        hub = max(strength, key=strength.get)
        insight(f"{len(ed)} links connect {G.number_of_nodes()} zones; the strongest corridor is "
                f"{located.loc[int(strongest['a']), 'Zone']} ↔ {located.loc[int(strongest['b']), 'Zone']} "
                f"({int(strongest['n'])} trips); density is {nx.density(G):.2f}, "
                f"{located.loc[hub, 'Zone']} is the biggest hub (weighted degree {strength[hub]:,}), "
                f"{located.loc[bridge, 'Zone']} is the strongest bridge (betweenness {betw[bridge]:.2f}), "
                f"and greedy modularity finds {len(comms)} communities (Q = {modularity:.2f}).")
        with st.expander("Network metrics per zone (top 10 by betweenness)"):
            tbl = nd.sort_values("betweenness", ascending=False).head(10)[
                ["Zone", "Borough", "trips", "strength", "betweenness", "community"]].rename(
                columns={"trips": "Trip ends", "strength": "Weighted degree",
                         "betweenness": "Betweenness", "community": "Community"})
            st.dataframe(tbl.round(3), hide_index=True, width="stretch")
    else:
        insight("No link with coordinates reaches the current minimum; lower the minimum-trips slider or widen the filters.")

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

@st.cache_data
def km_frame(data: pd.DataFrame):
    x = pd.DataFrame({"log_distance": np.log1p(data["trip_distance"]), "log_duration": np.log1p(data["duration_min"]),
                      "speed_mph": data["speed_mph"].clip(upper=60), "passengers": data["passenger_count"],
                      "hour_sin": np.sin(2 * np.pi * data["hour"] / 24), "hour_cos": np.cos(2 * np.pi * data["hour"] / 24)})
    return StandardScaler().fit_transform(x)

@st.cache_data
def run_kmeans(data: pd.DataFrame, k: int):
    Xs = km_frame(data)
    km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(Xs)
    sil = silhouette_score(Xs, km.labels_, sample_size=5000, random_state=0)
    return km.labels_, PCA(n_components=3, random_state=0).fit_transform(Xs), sil

@st.cache_data
def elbow(data: pd.DataFrame):
    Xs = km_frame(data)
    return [KMeans(n_clusters=k, n_init=3, random_state=42).fit(Xs).inertia_ for k in range(2, 9)]

RF_NUM = ["trip_distance", "duration_min", "speed_mph", "passenger_count", "hour", "is_airport"]
RF_CAT = ["pu_borough", "do_borough", "weekday"]

@st.cache_resource
def train_rf(data: pd.DataFrame):
    d = data[data["high_tip"].notna()].copy()
    d["weekday"] = d["weekday"].astype(str)
    X, y = d[RF_NUM + RF_CAT], d["high_tip"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=42)
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), RF_CAT)], remainder="passthrough")
    pipe = Pipeline([("pre", pre), ("rf", RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_leaf=20, class_weight="balanced", random_state=42, n_jobs=-1))])
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= .5).astype(int)
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["rf"].feature_importances_, index=[n.split("__")[-1] for n in names])
    imp.index = [next((c for c in RF_CAT if n.startswith(c)), n) for n in imp.index]

    cv = cross_val_score(pipe, Xtr, ytr, cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="roc_auc")
    base_pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), RF_CAT),
                                  ("num", StandardScaler(), RF_NUM)])
    lr = Pipeline([("pre", base_pre), ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))]).fit(Xtr, ytr)
    lr_auc = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
    extra = dict(cv_mean=float(cv.mean()), cv_std=float(cv.std()), lr_auc=float(lr_auc),
                 precision=float(precision_score(yte, pred)), recall=float(recall_score(yte, pred)))
    return (pipe, roc_auc_score(yte, proba), float((pred == yte).mean()), confusion_matrix(yte, pred),
            imp.groupby(level=0).sum().sort_values(), float(y.mean()), extra)

with tab4:
    st.info("ℹ️ **How the sidebar filters affect this tab:** the models are trained once on the full cleaned sample, so their "
            "results (clusters, silhouette, elbow curve, ROC-AUC, feature importance) do **not** change with the filters. "
            "Only the K-Means scatter and cluster profile show just the trips that match the current filters.")
    st.subheader("ML 1 · K-Means trip clustering (unsupervised)")
    ml_pool = df_all.sample(30000, random_state=7)
    k = st.slider("Number of clusters (k)", 2, 8, 4)
    labels, coords, sil = run_kmeans(ml_pool, k)
    ml = ml_pool.assign(cluster=pd.Categorical(["Cluster " + str(i + 1) for i in labels]),
                        pc1=coords[:, 0], pc2=coords[:, 1], pc3=coords[:, 2])
    ml = ml[ml.index.isin(df.index)] if len(df) < len(df_all) else ml
    if len(ml) < 50:
        st.info("Too few trips under the current filters for clustering.")
    else:
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = px.scatter_3d(ml.sample(min(3000, len(ml)), random_state=3), x="pc1", y="pc2", z="pc3",
                                color="cluster", opacity=.7, category_orders={"cluster": sorted(ml["cluster"].unique())},
                                hover_data=["trip_distance", "duration_min", "fare_amount", "pu_borough"],
                                title=f"K-Means (k={k}) in PCA space · silhouette = {sil:.2f}")
            fig.update_traces(marker_size=3)
            st.plotly_chart(style(fig, 520), width="stretch")
        with c2:
            fig = px.line(x=list(range(2, 9)), y=elbow(ml_pool.sample(10000, random_state=1)), markers=True,
                          title="Elbow curve (inertia)", labels={"x": "k", "y": "Inertia"})
            fig.add_vline(x=k, line_dash="dash", line_color="#E45756")
            st.plotly_chart(style(fig, 250, False), width="stretch")
            prof = ml.groupby("cluster", observed=True).agg(
                trips=("hour", "size"), distance_mi=("trip_distance", "mean"), minutes=("duration_min", "mean"),
                speed_mph=("speed_mph", "mean"), fare=("fare_amount", "mean"), passengers=("passenger_count", "mean"),
                airport_share=("is_airport", "mean")).round(2)
            st.markdown("**Cluster profile (means)**")
            st.dataframe(prof, width="stretch")
        p_long, p_grp, p_big = prof["distance_mi"].idxmax(), prof["passengers"].idxmax(), prof["trips"].idxmax()
        insight(f"{p_long} has the longest trips ({prof.loc[p_long, 'distance_mi']:.1f} mi, ${prof.loc[p_long, 'fare']:.0f} fare, "
                f"{prof.loc[p_long, 'airport_share']:.0%} airport share); {p_grp} has the most passengers "
                f"({prof.loc[p_grp, 'passengers']:.1f} on average); {p_big} is the largest group ({int(prof.loc[p_big, 'trips']):,} trips). "
                f"Silhouette = {sil:.2f}.")

    st.divider()
    st.subheader(f"ML 2 · Random-Forest classifier — will a card-paying rider tip ≥ {TIP_THRESHOLD} %?")
    pipe, auc, acc, cm, imp, base, ex = train_rf(df_all.sample(80000, random_state=11))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC (25 % hold-out)", f"{auc:.3f}")
    m2.metric("3-fold CV ROC-AUC", f"{ex['cv_mean']:.3f} ± {ex['cv_std']:.3f}")
    m3.metric("Baseline: logistic regression AUC", f"{ex['lr_auc']:.3f}", f"{auc - ex['lr_auc']:+.3f} for the forest", delta_color="off")
    m4.metric("Share of high tips", f"{base * 100:.1f} %")
    st.caption(f"At threshold 0.5 the forest reaches accuracy {acc:.3f}, precision {ex['precision']:.3f} and recall {ex['recall']:.3f} "
               f"for the 'high tip' class. The 'high tip' cut-off ({TIP_THRESHOLD} % of the fare) is a modelling choice.")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(imp, orientation="h", color_discrete_sequence=["#4C78A8"], title="Feature importance",
                     labels={"value": "Importance", "index": ""})
        st.plotly_chart(style(fig, 400, False), width="stretch")
    with c2:
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", x=["Low tip", "High tip"],
                        y=["Low tip", "High tip"], title="Confusion matrix (hold-out)",
                        labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(style(fig, 400, False), width="stretch")
    top2 = imp.sort_values(ascending=False).head(2)
    level = "strongly" if auc >= .8 else "partly" if auc >= .65 else "only weakly"
    insight(f"{top2.index[0]} and {top2.index[1]} are the strongest predictors of a high tip (importance {top2.iloc[0]:.2f} and "
            f"{top2.iloc[1]:.2f}). ROC-AUC = {auc:.2f} (CV {ex['cv_mean']:.2f}) vs {ex['lr_auc']:.2f} for a logistic baseline, "
            f"so tipping behaviour is {level} predictable.",
            scope="full sample, model ignores sidebar filters")

    st.markdown("##### 🔮 What-if: probability of a high tip")
    w1, w2, w3, w4 = st.columns(4)
    p_dist = w1.slider("Distance (miles)", 0.5, 25.0, 3.0, 0.5)
    p_hour = w2.slider("Pickup hour", 0, 23, 18)
    p_pass = w3.slider("Passengers", 1, 6, 1)
    p_air = w4.checkbox("Airport trip", value=False)
    b1, b2, b3 = st.columns(3)
    p_pu = b1.selectbox("Pickup borough", [b for b in boroughs_all if b not in ("Unknown", "N/A")], index=2 if len(boroughs_all) > 2 else 0)
    p_do = b2.selectbox("Dropoff borough", [b for b in boroughs_all if b not in ("Unknown", "N/A")], index=2 if len(boroughs_all) > 2 else 0)
    p_day = b3.selectbox("Weekday", DAYS, index=4)
    speed = float(df_all.loc[df_all["is_airport"] == int(p_air), "speed_mph"].median())
    row = pd.DataFrame([dict(trip_distance=p_dist, duration_min=p_dist / speed * 60, speed_mph=speed,
                             passenger_count=p_pass, hour=p_hour, is_airport=int(p_air),
                             pu_borough=p_pu, do_borough=p_do, weekday=p_day)])
    prob = pipe.predict_proba(row)[0, 1]
    gauge = go.Figure(go.Indicator(mode="gauge+number", value=prob * 100, number=dict(suffix=" %"),
                                   gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#4C78A8"),
                                              steps=[dict(range=[0, 33], color="#ffd6d6"),
                                                     dict(range=[33, 66], color="#fff3bf"),
                                                     dict(range=[66, 100], color="#d8f3dc")])))
    st.plotly_chart(style(gauge, 260, False), width="stretch")

with tab5:
    show = df.drop(columns=["value", "pickup", "date"])
    st.subheader("Filtered data (first 2,000 rows shown)")
    st.dataframe(show.head(2000), width="stretch", height=420)
    st.download_button("⬇️ Download filtered CSV", show.to_csv(index=False), "filtered_trips.csv", "text/csv")
    st.subheader("Summary statistics")
    st.dataframe(df[["trip_distance", "duration_min", "speed_mph", "fare_amount", "tip_amount", "total_amount",
                     "passenger_count"]].describe().round(2), width="stretch")
