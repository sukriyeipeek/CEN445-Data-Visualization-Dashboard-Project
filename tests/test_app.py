from pathlib import Path
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app.py")

@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(APP, default_timeout=900).run()

def data_table(app):
    return next(d.value for d in app.dataframe if "tip_pct" in d.value.columns)

def test_runs_without_exception(app):
    assert not app.exception

def test_every_chart_has_a_computed_insight(app):
    insights = [c.value for c in app.caption if "Insight" in c.value]
    assert len(insights) >= 11

def test_dataset_meets_assignment_size(app):
    table = data_table(app)
    assert len(table) >= 2000 and table.shape[1] >= 7

def test_cleaning_rules_hold(app):
    d = data_table(app)
    assert d["fare_amount"].between(0.01, 300).all()
    assert d["trip_distance"].between(0.01, 100).all()
    assert d["duration_min"].between(1, 180).all()
    assert (d["speed_mph"] <= 80).all()
    assert (d["passenger_count"] >= 1).all()

def test_tip_pct_only_for_recorded_card_tips(app):
    d = data_table(app)
    has_tip = d["tip_pct"].notna()
    assert (d.loc[has_tip, "payment_type"] == 1).all()
    assert (d.loc[has_tip, "RatecodeID"] != 99).all()
    assert d["tip_pct"].dropna().between(0, 100).all()
    assert not np.isnan(d.loc[d["payment_type"] != 1, "tip_pct"].fillna(0)).any()

@pytest.mark.parametrize("borough", [["Queens"], ["Staten Island"], ["Manhattan", "Brooklyn"]])
def test_filters_do_not_crash(borough):
    at = AppTest.from_file(APP, default_timeout=900).run()
    at.sidebar.multiselect[0].set_value(borough).run()
    assert not at.exception

def test_airport_only_and_revenue_metric_do_not_crash():
    at = AppTest.from_file(APP, default_timeout=900).run()
    at.sidebar.radio[0].set_value("Revenue (USD)")
    at.sidebar.checkbox[0].set_value(True).run()
    assert not at.exception
