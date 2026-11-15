import json
import sys
from pathlib import Path
import pandas as pd

trips_path, lookup_path, geo_path = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 150_000
out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)

trips = pd.read_parquet(trips_path)
print("raw trips:", trips.shape)
sample = trips.sample(n, random_state=445).sort_values("tpep_pickup_datetime").reset_index(drop=True)
sample.to_parquet(out / "yellow_taxi_2025_01_sample.parquet", index=False, compression="zstd")
print("sample:", sample.shape)

def ring_centroid(ring):
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        c = x0 * y1 - x1 * y0
        a += c
        cx += (x0 + x1) * c
        cy += (y0 + y1) * c
    a *= 0.5
    return (cx / (6 * a), cy / (6 * a), abs(a)) if a else (ring[0][0], ring[0][1], 0.0)

def rnd(c):
    return [rnd(i) for i in c] if isinstance(c[0], list) else [round(c[0], 5), round(c[1], 5)]

geo = json.load(open(geo_path, encoding="utf-8"))
lookup = pd.read_csv(lookup_path)
rows = []
for f in geo["features"]:
    g = f["geometry"]
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    g["coordinates"] = rnd(g["coordinates"])
    cx, cy, _ = max((ring_centroid([tuple(p) for p in poly[0]]) for poly in polys), key=lambda t: t[2])
    f["properties"] = {"locationid": int(f["properties"]["locationid"])}
    rows.append((f["properties"]["locationid"], cy, cx))
    f["id"] = f["properties"]["locationid"]
json.dump(geo, open(out / "taxi_zones.geojson", "w"), separators=(",", ":"))

cent = pd.DataFrame(rows, columns=["LocationID", "lat", "lon"]).drop_duplicates("LocationID")
zones = lookup.merge(cent, on="LocationID", how="left")
zones.to_csv(out / "zones.csv", index=False)
print("zones:", zones.shape, "without coordinates:", int(zones.lat.isna().sum()))
