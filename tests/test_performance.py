"""Performance tests for SophiaAgent research engines.

Uses time.monotonic() for wall-clock timing.  Tests are lenient to account
for CI variability; they should pass comfortably on modern hardware.
"""

import time

import numpy as np
import pandas as pd
import pytest

from sophia.research.causal import CausalEngine
from sophia.research.result_store import ResultStore
from sophia.research.statistics import StatisticalEngine


@pytest.fixture
def perf_store(tmp_path):
    s = ResultStore(str(tmp_path))
    yield s
    s.close()


class TestPerformanceResultStore:
    """ResultStore I/O performance."""

    def test_store_large_dataframe_under_1s(self, perf_store):
        df = pd.DataFrame(np.random.randn(50000, 20))
        t0 = time.monotonic()
        rid = perf_store.store(df, kind="dataframe", tool="research_test", params={})
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0, f"Store took {elapsed:.2f}s"

    def test_get_large_dataframe_under_1s(self, perf_store):
        df = pd.DataFrame(np.random.randn(50000, 20))
        rid = perf_store.store(df, kind="dataframe", tool="research_test", params={})
        t0 = time.monotonic()
        loaded = perf_store.get(rid)
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0, f"Get took {elapsed:.2f}s"
        assert loaded.shape == df.shape

    def test_metadata_lookup_under_50ms(self, perf_store):
        rid = perf_store.store({"x": 1}, kind="result", tool="research_test", params={})
        t0 = time.monotonic()
        meta = perf_store.get_metadata(rid)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.05, f"Metadata lookup took {elapsed:.3f}s"
        assert meta["tool"] == "research_test"

    def test_lineage_lookup_under_100ms(self, perf_store):
        rid1 = perf_store.store({"s": 1}, kind="result", tool="t1", params={})
        rid2 = perf_store.store({"s": 2}, kind="result", tool="t2", params={}, parents=[rid1])
        rid3 = perf_store.store({"s": 3}, kind="result", tool="t3", params={}, parents=[rid2])
        t0 = time.monotonic()
        lineage = perf_store.lineage(rid3)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1, f"Lineage took {elapsed:.3f}s"
        assert len(lineage) == 3


class TestPerformanceCausal:
    """Causal engine performance."""

    def test_did_10k_obs_under_5s(self, perf_store):
        engine = CausalEngine(store=perf_store)
        np.random.seed(42)
        n = 10000
        treat = np.random.binomial(1, 0.5, n)
        post = np.random.binomial(1, 0.5, n)
        y = 10 + 2 * treat + 1 * post + 3 * (treat * post) + np.random.normal(0, 1, n)

        t0 = time.monotonic()
        result = engine.did({
            "y": y.tolist(),
            "treat": treat.tolist(),
            "post": post.tolist(),
        })
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0, f"DID (N=10K) took {elapsed:.2f}s"
        assert "error" not in result.lower()

    def test_did_panel_50k_under_30s(self, perf_store):
        engine = CausalEngine(store=perf_store)
        np.random.seed(42)
        n_units = 500
        n_periods = 100
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)
        treat = (units >= 250).astype(float)
        post = (times >= 50).astype(float)
        y = (10.0
             + np.random.normal(0, 1, n_units)[units]
             + 0.05 * times
             + 2.0 * treat * post
             + np.random.normal(0, 1, len(units)))

        t0 = time.monotonic()
        result = engine.did({
            "y": y.tolist(),
            "treat": treat.tolist(),
            "post": post.tolist(),
            "unit": units.tolist(),
            "time": times.tolist(),
        })
        elapsed = time.monotonic() - t0
        assert elapsed < 30.0, f"Panel DID (N=50K) took {elapsed:.2f}s"
        assert "error" not in result.lower()

    def test_scm_30periods_20donors_under_10s(self, perf_store):
        engine = CausalEngine(store=perf_store)
        np.random.seed(42)
        years = list(range(1960, 1990))
        regions = ["Treated"] + [f"Donor_{i}" for i in range(19)]
        rows = []
        for r in regions:
            base = np.random.uniform(2.0, 4.0)
            for y in years:
                rows.append({
                    "region": r,
                    "year": y,
                    "gdp": base + 0.05 * (y - 1960) + np.random.normal(0, 0.1),
                })
        df = pd.DataFrame(rows)

        t0 = time.monotonic()
        result = engine.synthetic_control({
            "y": df["gdp"].tolist(),
            "unit": df["region"].tolist(),
            "time": df["year"].tolist(),
            "treated_unit": "Treated",
            "treatment_time": 1980,
        })
        elapsed = time.monotonic() - t0
        assert elapsed < 10.0, f"SCM (T=30, J=20) took {elapsed:.2f}s"
        assert "error" not in result.lower()


class TestPerformanceStatistics:
    """Statistical engine performance."""

    def test_ttest_1million_under_2s(self, perf_store):
        engine = StatisticalEngine(store=perf_store)
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 500000).tolist()
        g2 = np.random.normal(0.5, 1, 500000).tolist()

        t0 = time.monotonic()
        result = engine.ttest({"group1": g1, "group2": g2})
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, f"t-test (N=1M) took {elapsed:.2f}s"
        assert "error" not in result.lower()
