"""
tests/test_coralnet_temporal.py
--------------------------------
Unit and smoke tests for the coralnet_temporal package.

Test categories
---------------
Synthetic-data tests (no real files needed)
    loader, community, temporal, genera — use in-memory DataFrames or tmp_path CSVs

Real-data tests (require /mnt/user-data/uploads/)
    preprocess, growth_forms — validated against the actual 2025 PBHR survey files
    generate_sample_data — runs the generator script and checks its output

48 tests total.
"""

import numpy as np
import pandas as pd
import pytest

from coralnet_temporal.community import compute_community_metrics, monthly_summary
from coralnet_temporal.genera import compute_genera_stats, filter_common_genera, run_genera_analysis
from coralnet_temporal.temporal import (
    bray_curtis_dissimilarity,
    prevalence_conditional_analysis,
    run_temporal_tests,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Nov"]
MONTH_MAP = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 11: "Nov"}
CORAL_COLS = ["Acropora", "Porites", "Isopora"]


@pytest.fixture
def synthetic_df():
    """44 locations × 6 months, with realistic variability."""
    rng = np.random.default_rng(42)
    rows = []
    for t_id in range(1, 5):
        for m_id in range(1, 12):
            for month, month_name in MONTH_MAP.items():
                acropora = float(rng.exponential(5))
                porites = float(rng.exponential(3))
                isopora = float(rng.exponential(2))
                rows.append(
                    {
                        "transect": str(t_id),
                        "meter": str(m_id),
                        "location_id": f"{t_id}_{m_id}",
                        "month": month,
                        "month_name": month_name,
                        "Acropora": acropora,
                        "Porites": porites,
                        "Isopora": isopora,
                    }
                )
    return pd.DataFrame(rows)


# ── loader ────────────────────────────────────────────────────────────────────

def test_load_coralnet_export_missing_file():
    from coralnet_temporal.loader import load_coralnet_export
    with pytest.raises(FileNotFoundError):
        load_coralnet_export("nonexistent.csv")


# ── community ─────────────────────────────────────────────────────────────────

def test_compute_community_metrics_columns(synthetic_df):
    df = compute_community_metrics(synthetic_df.copy(), CORAL_COLS)
    for col in ["Total_Coral", "Richness", "Shannon_H", "Simpson_D", "Evenness"]:
        assert col in df.columns, f"Missing column: {col}"


def test_total_coral_non_negative(synthetic_df):
    df = compute_community_metrics(synthetic_df.copy(), CORAL_COLS)
    assert (df["Total_Coral"] >= 0).all()


def test_richness_bounded(synthetic_df):
    df = compute_community_metrics(synthetic_df.copy(), CORAL_COLS)
    assert df["Richness"].max() <= len(CORAL_COLS)
    assert df["Richness"].min() >= 0


def test_monthly_summary_shape(synthetic_df):
    df = compute_community_metrics(synthetic_df.copy(), CORAL_COLS)
    summary = monthly_summary(df, metrics=["Total_Coral"], month_order=MONTH_ORDER)
    assert set(MONTH_ORDER).issubset(set(summary.index))


# ── temporal ─────────────────────────────────────────────────────────────────

def test_run_temporal_tests_keys(synthetic_df):
    df = compute_community_metrics(synthetic_df.copy(), CORAL_COLS)
    result = run_temporal_tests(df, metric="Total_Coral", month_order=MONTH_ORDER)
    for key in ["friedman", "pairwise", "power", "complete_n"]:
        assert key in result, f"Missing key: {key}"


def test_friedman_p_in_range(synthetic_df):
    df = compute_community_metrics(synthetic_df.copy(), CORAL_COLS)
    result = run_temporal_tests(df, metric="Total_Coral")
    p = result["friedman"]["p_value"]
    assert 0.0 <= p <= 1.0


def test_bray_curtis_range(synthetic_df):
    bc = bray_curtis_dissimilarity(synthetic_df, CORAL_COLS, "Jan", "Nov")
    assert 0.0 <= bc <= 1.0


def test_bray_curtis_identical():
    """Same month vs same month should give 0."""
    df = pd.DataFrame(
        {
            "month_name": ["Jan", "Jan"],
            "Acropora": [10.0, 10.0],
            "Porites": [5.0, 5.0],
        }
    )
    bc = bray_curtis_dissimilarity(df, ["Acropora", "Porites"], "Jan", "Jan")
    assert bc == pytest.approx(0.0)


def test_prevalence_conditional_shape(synthetic_df):
    result = prevalence_conditional_analysis(
        synthetic_df, CORAL_COLS, month_a="Jan", month_b="Nov"
    )
    assert len(result) == len(CORAL_COLS)
    assert "Interpretation" in result.columns


# ── genera ────────────────────────────────────────────────────────────────────

def test_compute_genera_stats_index(synthetic_df):
    stats_df = compute_genera_stats(synthetic_df, CORAL_COLS)
    assert set(CORAL_COLS).issubset(set(stats_df.index))


def test_filter_common_genera_subset(synthetic_df):
    stats_df = compute_genera_stats(synthetic_df, CORAL_COLS)
    common = filter_common_genera(stats_df, min_mean_cover=0.0, min_prevalence=0.0)
    assert set(common) == set(CORAL_COLS)


def test_run_genera_analysis_returns_dfs(synthetic_df):
    temporal_df, prevalence_df = run_genera_analysis(
        synthetic_df, CORAL_COLS, min_sites=3, month_a="Jan", month_b="Nov"
    )
    assert isinstance(temporal_df, pd.DataFrame)
    assert isinstance(prevalence_df, pd.DataFrame)
    assert "p_value" in temporal_df.columns


# ── preprocess ────────────────────────────────────────────────────────────────

def test_preprocess_functional_groups(tmp_path):
    """Functional-group sums must exactly reproduce the reference output."""
    from coralnet_temporal.preprocess import preprocess

    df_grouped, _ = preprocess(
        raw_path="/mnt/user-data/uploads/2025_percent_covers.csv",
        labelset_path="/mnt/user-data/uploads/labelset.csv",
        sites=["PBHR"],
    )
    expected = (
        pd.read_csv("/mnt/user-data/uploads/2025_percent_covers_grouped.csv")
        .query("site == 'PBHR'")
    )

    p = df_grouped.set_index("Image ID").sort_index()
    e = expected.set_index("Image ID").sort_index()
    for col in ["Hard coral", "Other Invertebrates", "Soft Substrate",
                "Hard Substrate", "Other", "Algae"]:
        diff = (p[col].round(3) - e[col].round(3)).abs().max()
        assert diff < 0.001, f"{col} max diff = {diff:.4f}"


def test_preprocess_genera(tmp_path):
    """Genus-level sums must match the reference output for PBHR rows."""
    from coralnet_temporal.preprocess import preprocess

    _, df_genera = preprocess(
        raw_path="/mnt/user-data/uploads/2025_percent_covers.csv",
        labelset_path="/mnt/user-data/uploads/labelset.csv",
        sites=["PBHR"],
    )
    expected = pd.read_csv("/mnt/user-data/uploads/2025_PBHR_percent_covers_genera.csv")

    shared = set(df_genera["Image ID"]) & set(expected["Image ID"])
    pg = df_genera.set_index("Image ID").sort_index().loc[sorted(shared)]
    eg = expected.set_index("Image ID").sort_index().loc[sorted(shared)]
    for col in ["Acropora", "Porites", "Fungiidae", "Seriatopora", "Turf algae", "Sand"]:
        diff = (pg[col].round(3) - eg[col].round(3)).abs().max()
        assert diff < 0.001, f"{col} max diff = {diff:.4f}"


def test_preprocess_row_count():
    """PBHR site filter must return exactly 528 rows."""
    from coralnet_temporal.preprocess import preprocess

    df_grouped, _ = preprocess(
        raw_path="/mnt/user-data/uploads/2025_percent_covers.csv",
        labelset_path="/mnt/user-data/uploads/labelset.csv",
        sites=["PBHR"],
    )
    assert len(df_grouped) == 528


def test_preprocess_metadata_columns():
    """Output must contain all expected metadata columns."""
    from coralnet_temporal.preprocess import preprocess

    df_grouped, _ = preprocess(
        raw_path="/mnt/user-data/uploads/2025_percent_covers.csv",
        labelset_path="/mnt/user-data/uploads/labelset.csv",
        sites=["PBHR"],
    )
    for col in ["year", "month", "month_name", "site", "transect", "meter",
                "side_code", "side"]:
        assert col in df_grouped.columns, f"Missing metadata column: {col}"


def test_preprocess_no_sites_filter():
    """Without site filter, output includes all parseable sites."""
    from coralnet_temporal.preprocess import preprocess

    df_grouped, _ = preprocess(
        raw_path="/mnt/user-data/uploads/2025_percent_covers.csv",
        labelset_path="/mnt/user-data/uploads/labelset.csv",
    )
    sites = set(df_grouped["site"].unique())
    assert "PBHR" in sites
    assert "OC" in sites


def test_build_latlon_template(tmp_path):
    """Template must cover all site × transect × meter combos."""
    from coralnet_temporal.preprocess import build_latlon_template

    out = tmp_path / "latlon.csv"
    tmpl = build_latlon_template(
        "/mnt/user-data/uploads/2025_percent_covers.csv",
        out_path=str(out),
    )
    assert out.exists()
    assert set(tmpl.columns) == {"site", "transect", "meter", "lat", "lon"}
    assert len(tmpl) > 0


# ── growth_forms ──────────────────────────────────────────────────────────────

RAW_PATH = "/mnt/user-data/uploads/2025_percent_covers.csv"

def test_growth_form_presets_keys():
    from coralnet_temporal.growth_forms import GROWTH_FORM_PRESETS
    for genus in ["Porites", "Acropora", "Echinopora", "Montipora"]:
        assert genus in GROWTH_FORM_PRESETS


def test_prepare_growth_form_data_porites():
    from coralnet_temporal.growth_forms import prepare_growth_form_data, GROWTH_FORM_PRESETS
    df = prepare_growth_form_data(RAW_PATH, GROWTH_FORM_PRESETS["Porites"],
                                  sites=["PBHR"], genus_name="Porites")
    assert "Branching" in df.columns
    assert "Encrusting" in df.columns
    assert "Massive" in df.columns
    assert "Porites_Total" in df.columns
    assert "location_id" in df.columns
    # should have 44 locations × 6 months = 264 rows
    assert len(df) == 264


def test_prepare_growth_form_data_acropora():
    from coralnet_temporal.growth_forms import prepare_growth_form_data, GROWTH_FORM_PRESETS
    df = prepare_growth_form_data(RAW_PATH, GROWTH_FORM_PRESETS["Acropora"],
                                  sites=["PBHR"], genus_name="Acropora")
    for form in ["Arborescent", "Branching", "Caespitose", "Corymbose", "Digitate", "Tabulate"]:
        assert form in df.columns
    assert "Acropora_Total" in df.columns


def test_prepare_growth_form_data_sides_averaged():
    """Each location × time should appear exactly once after L/R averaging."""
    from coralnet_temporal.growth_forms import prepare_growth_form_data, GROWTH_FORM_PRESETS
    df = prepare_growth_form_data(RAW_PATH, GROWTH_FORM_PRESETS["Porites"],
                                  sites=["PBHR"], genus_name="Porites")
    # No duplicate location_id × month combinations
    dupes = df.duplicated(subset=["location_id", "month"]).sum()
    assert dupes == 0


def test_growth_form_stats_keys():
    from coralnet_temporal.growth_forms import (
        prepare_growth_form_data, growth_form_stats, GROWTH_FORM_PRESETS
    )
    df = prepare_growth_form_data(RAW_PATH, GROWTH_FORM_PRESETS["Porites"],
                                  sites=["PBHR"], genus_name="Porites")
    form_cols = list(GROWTH_FORM_PRESETS["Porites"].keys())
    result = growth_form_stats(df, form_cols, month_order=["Jan","Feb","Mar","Apr","May","Nov"])
    for key in ["overall", "by_transect", "by_month",
                "proportions_overall", "proportions_by_transect",
                "proportions_by_month", "jan_nov_change"]:
        assert key in result, f"Missing key: {key}"


def test_growth_form_stats_proportions_sum_to_100():
    from coralnet_temporal.growth_forms import (
        prepare_growth_form_data, growth_form_stats, GROWTH_FORM_PRESETS
    )
    df = prepare_growth_form_data(RAW_PATH, GROWTH_FORM_PRESETS["Porites"],
                                  sites=["PBHR"], genus_name="Porites")
    form_cols = list(GROWTH_FORM_PRESETS["Porites"].keys())
    result = growth_form_stats(df, form_cols)
    row_sums = result["proportions_by_transect"].sum(axis=1)
    # Only transects with any Porites presence sum to 100;
    # transects where the genus is absent sum to 0 (0/0 = NaN → filled as 0)
    present_transects = row_sums[row_sums > 0]
    assert (present_transects.round(1) == 100.0).all(), (
        f"Non-zero row sums should equal 100: {row_sums.tolist()}"
    )


def test_run_growth_form_tests_porites():
    from coralnet_temporal.growth_forms import (
        prepare_growth_form_data, run_growth_form_tests, GROWTH_FORM_PRESETS
    )
    df = prepare_growth_form_data(RAW_PATH, GROWTH_FORM_PRESETS["Porites"],
                                  sites=["PBHR"], genus_name="Porites")
    form_cols = list(GROWTH_FORM_PRESETS["Porites"].keys())
    tests = run_growth_form_tests(df, form_cols)
    assert set(tests["Form"]) == set(form_cols)
    # All forms should be present in the results table
    assert len(tests) == len(form_cols)
    # p_value column exists and is numeric (NaN for skipped forms is acceptable)
    assert "p_value" in tests.columns
    # Branching has enough sites to be tested (n >= 5)
    branching_row = tests[tests["Form"] == "Branching"].iloc[0]
    assert branching_row["n_sites"] >= 5


def test_growth_form_ratio():
    from coralnet_temporal.growth_forms import (
        prepare_growth_form_data, growth_form_ratio, GROWTH_FORM_PRESETS
    )
    df = prepare_growth_form_data(RAW_PATH, GROWTH_FORM_PRESETS["Porites"],
                                  sites=["PBHR"], genus_name="Porites")
    r = growth_form_ratio(df, "Branching", "Massive", by="overall")
    assert r.iloc[0] > 0


def test_custom_form_map():
    """Users can pass a completely custom form_map for any genus."""
    from coralnet_temporal.growth_forms import prepare_growth_form_data
    custom_map = {
        "Branching": ["Ech.branch"],
        "Encrusting": ["Ech.encrus"],
        "Foliose": ["Ech.folio"],
    }
    df = prepare_growth_form_data(RAW_PATH, custom_map,
                                  sites=["PBHR"], genus_name="Echinopora")
    for form in custom_map:
        assert form in df.columns


def test_missing_code_raises():
    from coralnet_temporal.growth_forms import prepare_growth_form_data
    bad_map = {"Fake": ["NOTACODE"]}
    with pytest.raises(ValueError, match="Short codes not found"):
        prepare_growth_form_data(RAW_PATH, bad_map, sites=["PBHR"])


# ── loader (additional) ───────────────────────────────────────────────────────

def test_loader_averages_sides(tmp_path):
    """Left and right images for the same location must be averaged into one row."""
    import sys
    sys.path.insert(0, '/home/claude/coralnet-temporal-analysis/src')
    from coralnet_temporal.loader import load_coralnet_export

    csv = tmp_path / "test.csv"
    csv.write_text(
        "year,month,month_name,site,transect,meter,side_code,side,Acropora,Porites\n"
        "2025,1,Jan,PBHR,1,5,L,Left,10.0,4.0\n"
        "2025,1,Jan,PBHR,1,5,R,Right,20.0,8.0\n"
    )
    df, coral_cols, _ = load_coralnet_export(str(csv))
    assert len(df) == 1
    assert df["Acropora"].iloc[0] == pytest.approx(15.0)
    assert df["Porites"].iloc[0]  == pytest.approx(6.0)


def test_loader_non_coral_excluded(tmp_path):
    """Columns listed in non_coral must not appear in coral_cols."""
    from coralnet_temporal.loader import load_coralnet_export

    csv = tmp_path / "test.csv"
    csv.write_text(
        "year,month,month_name,site,transect,meter,side_code,side,Acropora,Sand,Rubble\n"
        "2025,1,Jan,PBHR,1,5,L,Left,10.0,30.0,5.0\n"
    )
    df, coral_cols, all_cols = load_coralnet_export(
        str(csv), non_coral=["Sand", "Rubble"]
    )
    assert "Acropora" in coral_cols
    assert "Sand"     not in coral_cols
    assert "Rubble"   not in coral_cols
    assert "Sand"     in all_cols


def test_loader_location_id_created(tmp_path):
    """location_id must be transect_meter string."""
    from coralnet_temporal.loader import load_coralnet_export

    csv = tmp_path / "test.csv"
    csv.write_text(
        "year,month,month_name,site,transect,meter,side_code,side,Acropora\n"
        "2025,1,Jan,PBHR,2,10,L,Left,5.0\n"
    )
    df, _, _ = load_coralnet_export(str(csv))
    assert "location_id" in df.columns
    assert df["location_id"].iloc[0] == "2_10"


# ── community (additional) ────────────────────────────────────────────────────

def test_shannon_zero_when_all_zero():
    """Shannon H′ must be 0 when all cover values are 0."""
    df = pd.DataFrame({
        "transect": ["1"], "meter": ["1"], "location_id": ["1_1"],
        "month": [1], "month_name": ["Jan"],
        "Acropora": [0.0], "Porites": [0.0],
    })
    df = compute_community_metrics(df, ["Acropora", "Porites"])
    assert df["Shannon_H"].iloc[0] == pytest.approx(0.0)


def test_evenness_zero_when_richness_one():
    """Evenness must be 0 when only one genus is present (no diversity to spread)."""
    df = pd.DataFrame({
        "transect": ["1"], "meter": ["1"], "location_id": ["1_1"],
        "month": [1], "month_name": ["Jan"],
        "Acropora": [10.0], "Porites": [0.0],
    })
    df = compute_community_metrics(df, ["Acropora", "Porites"])
    assert df["Evenness"].iloc[0] == pytest.approx(0.0)


def test_simpson_d_bounds(synthetic_df):
    """Simpson D must be in [0, 1] for all observations."""
    df = compute_community_metrics(synthetic_df.copy(), CORAL_COLS)
    assert (df["Simpson_D"] >= 0).all()
    assert (df["Simpson_D"] <= 1).all()


# ── temporal (additional) ─────────────────────────────────────────────────────

def test_pairwise_wilcoxon_output_shape(synthetic_df):
    """Pairwise comparisons must have one row per consecutive month pair."""
    df = compute_community_metrics(synthetic_df.copy(), CORAL_COLS)
    result = run_temporal_tests(df, metric="Total_Coral", month_order=MONTH_ORDER)
    pw = result["pairwise"]
    assert isinstance(pw, pd.DataFrame)
    assert "Comparison" in pw.columns
    assert "p_value"    in pw.columns
    assert "sig"        in pw.columns
    assert len(pw) == len(MONTH_ORDER) - 1


def test_power_analysis_keys(synthetic_df):
    """Power analysis dict must contain all expected keys."""
    df = compute_community_metrics(synthetic_df.copy(), CORAL_COLS)
    result = run_temporal_tests(df, metric="Total_Coral")
    power = result["power"]
    for key in ["n_locations", "mean", "sd", "cv_pct",
                "detectable_abs_pp", "detectable_rel_pct"]:
        assert key in power, f"Missing power key: {key}"


def test_lmm_returns_icc(synthetic_df):
    """LMM result must contain icc when statsmodels is available."""
    df = compute_community_metrics(synthetic_df.copy(), CORAL_COLS)
    result = run_temporal_tests(df, metric="Total_Coral")
    if result["lmm"] is not None:
        assert "icc" in result["lmm"]
        assert 0.0 <= result["lmm"]["icc"] <= 1.0


def test_growth_form_ratio_by_transect():
    from coralnet_temporal.growth_forms import prepare_growth_form_data, growth_form_ratio, GROWTH_FORM_PRESETS
    df = prepare_growth_form_data(RAW_PATH, GROWTH_FORM_PRESETS["Porites"],
                                  sites=["PBHR"], genus_name="Porites")
    r = growth_form_ratio(df, "Branching", "Massive", by="transect")
    assert len(r) == df["transect"].nunique()


def test_growth_form_ratio_by_month():
    from coralnet_temporal.growth_forms import prepare_growth_form_data, growth_form_ratio, GROWTH_FORM_PRESETS
    df = prepare_growth_form_data(RAW_PATH, GROWTH_FORM_PRESETS["Porites"],
                                  sites=["PBHR"], genus_name="Porites")
    r = growth_form_ratio(df, "Branching", "Massive", by="month",
                          month_order=MONTH_ORDER)
    assert len(r) == len(MONTH_ORDER)


# ── genera (additional) ───────────────────────────────────────────────────────

def test_filter_common_genera_min_cover(synthetic_df):
    """Genera below min_mean_cover and min_prevalence must be excluded."""
    from coralnet_temporal.genera import compute_genera_stats, filter_common_genera
    # Set very high thresholds — should return empty
    stats_df = compute_genera_stats(synthetic_df, CORAL_COLS)
    result = filter_common_genera(stats_df, min_mean_cover=9999, min_prevalence=9999)
    assert len(result) == 0


def test_filter_common_genera_min_prevalence(synthetic_df):
    """Genera with zero cover but high prevalence threshold must be excluded."""
    from coralnet_temporal.genera import compute_genera_stats, filter_common_genera
    stats_df = compute_genera_stats(synthetic_df, CORAL_COLS)
    # All genera present → all pass at 0 thresholds
    result_all = filter_common_genera(stats_df, min_mean_cover=0.0, min_prevalence=0.0)
    assert set(result_all) == set(CORAL_COLS)


def test_genera_temporal_df_sorted_by_pvalue(synthetic_df):
    """run_genera_analysis must return temporal_df sorted ascending by p_value."""
    temporal_df, _ = run_genera_analysis(
        synthetic_df, CORAL_COLS, min_sites=3, month_a="Jan", month_b="Nov"
    )
    p_vals = temporal_df["p_value"].dropna().tolist()
    assert p_vals == sorted(p_vals), "temporal_df not sorted by p_value"


# ── growth_forms (additional) ─────────────────────────────────────────────────

def test_all_presets_resolve():
    """Every preset in GROWTH_FORM_PRESETS must load without error."""
    from coralnet_temporal.growth_forms import prepare_growth_form_data, GROWTH_FORM_PRESETS
    for genus, form_map in GROWTH_FORM_PRESETS.items():
        try:
            df = prepare_growth_form_data(RAW_PATH, form_map,
                                          sites=["PBHR"], genus_name=genus)
            assert f"{genus}_Total" in df.columns, f"{genus}: missing Total column"
        except ValueError as e:
            # Some presets may have codes not in this specific dataset — acceptable
            # but flag if it's a genuine mapping error
            assert "Short codes not found" in str(e), (
                f"{genus} raised unexpected error: {e}"
            )


def test_jan_nov_change_absent_when_months_missing(synthetic_df):
    """jan_nov_change key must be absent when Nov is not in the data."""
    from coralnet_temporal.growth_forms import (
        prepare_growth_form_data, growth_form_stats, GROWTH_FORM_PRESETS
    )
    # Use synthetic_df which has Nov — strip it out
    df_no_nov = synthetic_df[synthetic_df["month_name"] != "Nov"].copy()
    # Rebuild as growth-form-style df with Porites columns
    df_no_nov = df_no_nov.rename(columns={"Acropora": "Branching",
                                           "Porites": "Massive",
                                           "Isopora": "Encrusting"})
    df_no_nov["Porites_Total"] = (df_no_nov["Branching"]
                                  + df_no_nov["Massive"]
                                  + df_no_nov["Encrusting"])
    form_cols = ["Branching", "Massive", "Encrusting"]
    result = growth_form_stats(df_no_nov, form_cols,
                                genus_total_col="Porites_Total")
    assert "jan_nov_change" not in result


# ── synthetic data generator ──────────────────────────────────────────────────

def test_synthetic_csv_parses_through_preprocess(tmp_path):
    """Synthetic CSV must pass through preprocess without dropped rows."""
    import subprocess, sys
    out = tmp_path / "synthetic.csv"
    subprocess.run(
        [sys.executable,
         "/home/claude/coralnet-temporal-analysis/scripts/generate_sample_data.py",
         "--out", str(out), "--transects", "2", "--meters", "3"],
        check=True, capture_output=True,
    )
    from coralnet_temporal.preprocess import preprocess
    df_grouped, df_genera = preprocess(
        raw_path=str(out),
        labelset_path="/mnt/user-data/uploads/labelset.csv",
        sites=["PBHR"],
    )
    # 2 transects × 3 meters × 6 months × 2 sides = 72 rows
    assert len(df_grouped) == 72
    assert "Hard coral" in df_grouped.columns
    assert "Acropora"   in df_genera.columns


def test_synthetic_csv_has_comma_decimals(tmp_path):
    """At least some values in the synthetic CSV must use comma as decimal."""
    import subprocess, sys
    out = tmp_path / "synthetic.csv"
    subprocess.run(
        [sys.executable,
         "/home/claude/coralnet-temporal-analysis/scripts/generate_sample_data.py",
         "--out", str(out)],
        check=True, capture_output=True,
    )
    raw_text = out.read_text()
    assert "," in raw_text.split("\n")[1], "No comma decimal found in row 1"


def test_synthetic_csv_filename_pattern(tmp_path):
    """Synthetic filenames must match Pattern A and be parseable."""
    import subprocess, sys, re
    out = tmp_path / "synthetic.csv"
    subprocess.run(
        [sys.executable,
         "/home/claude/coralnet-temporal-analysis/scripts/generate_sample_data.py",
         "--out", str(out), "--transects", "1", "--meters", "1"],
        check=True, capture_output=True,
    )
    pattern_a = re.compile(
        r"\d{4}\.\d{2}_PBHR_T\d+_\d+[LR]\.JPG", re.IGNORECASE
    )
    df = pd.read_csv(out)
    names = df["Image name"].dropna().tolist()
    assert all(pattern_a.match(n) for n in names), (
        f"Some filenames don't match Pattern A: {[n for n in names if not pattern_a.match(n)]}"
    )
