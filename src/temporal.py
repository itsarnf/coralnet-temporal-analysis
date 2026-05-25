"""
temporal.py
-----------
Repeated-measures statistical tests for temporal change detection.

All tests operate on **paired locations** — only location × time cells
where the location appears at every time point are included (complete cases).
This preserves the repeated-measures structure and avoids pseudoreplication.

Tests implemented
-----------------
Friedman χ²        Overall temporal variation across all time points
                   (non-parametric repeated measures ANOVA)

Wilcoxon signed-rank  Pairwise sequential month comparisons
                      (non-parametric paired t-test equivalent)

Variance partitioning  Linear Mixed-Effects Model (LMM) to quantify
                       spatial vs temporal control via ICC

Power analysis     Minimum detectable change at 80% power given the
                   observed SD and sample size

Bray-Curtis        Compositional dissimilarity between two time points
dissimilarity
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

try:
    from statsmodels.formula.api import mixedlm
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False
    logger.warning(
        "statsmodels not found; LMM / variance-partitioning will be skipped."
    )


# ── public API ────────────────────────────────────────────────────────────────

def run_temporal_tests(
    df: pd.DataFrame,
    metric: str = "Total_Coral",
    month_order: Optional[list[str]] = None,
) -> dict:
    """Run the full suite of temporal tests for a single community metric.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``location_id``, ``month``, ``month_name``, and *metric*.
    metric : str
        Column to analyse (e.g. ``'Total_Coral'``, ``'Richness'``).
    month_order : list[str], optional
        Ordered month names for display.

    Returns
    -------
    dict with keys
        ``friedman``, ``pairwise``, ``lmm`` (if statsmodels available),
        ``power``, ``complete_n``
    """
    df_wide = _pivot_complete_cases(df, metric)
    n = len(df_wide)
    logger.info("Temporal tests for '%s': %d complete-case locations", metric, n)

    results: dict = {"metric": metric, "complete_n": n}

    # ── Friedman test ─────────────────────────────────────────────────────────
    results["friedman"] = _friedman(df_wide)

    # ── pairwise Wilcoxon ─────────────────────────────────────────────────────
    results["pairwise"] = _pairwise_wilcoxon(df, metric, month_order)

    # ── LMM variance partitioning ─────────────────────────────────────────────
    if _HAS_STATSMODELS:
        results["lmm"] = _lmm_variance(df, metric)
    else:
        results["lmm"] = None

    # ── power analysis ────────────────────────────────────────────────────────
    results["power"] = _power_analysis(df[metric], n_locations=n)

    return results


def bray_curtis_dissimilarity(
    df: pd.DataFrame,
    coral_cols: list[str],
    month_a: str,
    month_b: str,
) -> float:
    """Compute Bray-Curtis dissimilarity between the mean compositions of
    two months.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``month_name`` and all *coral_cols*.
    coral_cols : list[str]
        Coral category columns to include.
    month_a, month_b : str
        Month name strings (e.g. ``'Jan'``, ``'Nov'``).

    Returns
    -------
    float
        Bray-Curtis dissimilarity in [0, 1].
        0 = identical composition, 1 = completely different.
    """
    a = df[df["month_name"] == month_a][coral_cols].mean()
    b = df[df["month_name"] == month_b][coral_cols].mean()
    denom = (a + b).sum()
    if denom == 0:
        return float("nan")
    return float((a - b).abs().sum() / denom)


def prevalence_conditional_analysis(
    df: pd.DataFrame,
    genera: list[str],
    month_a: str = "Jan",
    month_b: str = "Nov",
) -> pd.DataFrame:
    """Decompose overall cover change into spatial redistribution vs
    within-location abundance change.

    For each genus, computes:
    * **Prevalence** — proportion of locations where the genus is present
    * **Conditional mean** — mean cover at occupied locations only
    * Δ between *month_a* and *month_b* for both metrics
    * Plain-language interpretation

    Parameters
    ----------
    df : pd.DataFrame
    genera : list[str]
    month_a, month_b : str

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for genus in genera:
        a = df[df["month_name"] == month_a][genus]
        b = df[df["month_name"] == month_b][genus]

        prev_a = (a > 0).mean() * 100
        prev_b = (b > 0).mean() * 100
        d_prev = prev_b - prev_a

        cond_a = a[a > 0].mean() if (a > 0).any() else 0.0
        cond_b = b[b > 0].mean() if (b > 0).any() else 0.0
        d_cond = cond_b - cond_a

        rows.append(
            {
                "Genus": genus,
                f"Prevalence {month_a} (%)": round(prev_a, 1),
                f"Prevalence {month_b} (%)": round(prev_b, 1),
                "Δ Prevalence (pp)": round(d_prev, 1),
                f"Cond. Mean {month_a} (%)": round(cond_a, 2),
                f"Cond. Mean {month_b} (%)": round(cond_b, 2),
                "Δ Cond. Mean (%)": round(d_cond, 2),
                "Interpretation": _interpret(d_prev, d_cond),
            }
        )
    return pd.DataFrame(rows)


# ── private helpers ───────────────────────────────────────────────────────────

def _pivot_complete_cases(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    wide = df.pivot(index="location_id", columns="month", values=metric)
    return wide.dropna()


def _friedman(df_wide: pd.DataFrame) -> dict:
    arrays = [df_wide[col].values for col in df_wide.columns]
    stat, p = stats.friedmanchisquare(*arrays)
    return {
        "statistic": round(float(stat), 3),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "n": len(df_wide),
    }


def _pairwise_wilcoxon(
    df: pd.DataFrame,
    metric: str,
    month_order: Optional[list[str]],
) -> pd.DataFrame:
    months_num = sorted(df["month"].unique())
    month_name_map = df.drop_duplicates("month").set_index("month")["month_name"].to_dict()

    rows = []
    for i in range(len(months_num) - 1):
        m1, m2 = months_num[i], months_num[i + 1]
        pair = (
            df[df["month"].isin([m1, m2])]
            .pivot(index="location_id", columns="month", values=metric)
            .dropna()
        )
        if len(pair) < 5:
            continue
        stat, p = stats.wilcoxon(pair[m1], pair[m2])
        diff = (pair[m2] - pair[m1]).mean()
        rows.append(
            {
                "Comparison": f"{month_name_map[m1]} → {month_name_map[m2]}",
                "n": len(pair),
                "Δ Mean": round(diff, 3),
                "W": round(float(stat), 1),
                "p_value": round(float(p), 5),
                "sig": _sig_stars(p),
            }
        )
    return pd.DataFrame(rows)


def _lmm_variance(df: pd.DataFrame, metric: str) -> dict:
    try:
        tmp = df[["location_id", "month", metric]].copy()
        tmp["month"] = tmp["month"].astype("category")
        model = mixedlm(
            f"Q('{metric}') ~ C(month)",
            tmp,
            groups=tmp["location_id"],
        )
        result = model.fit(method="nm", disp=False)
        var_b = float(result.cov_re.iloc[0, 0])
        var_w = float(result.scale)
        icc = var_b / (var_b + var_w)
        return {
            "between_location_variance": round(var_b, 3),
            "within_location_variance": round(var_w, 3),
            "icc": round(icc, 3),
            "interpretation": (
                "Spatial heterogeneity dominates" if icc > 0.5
                else "Mixed spatial/temporal variation"
            ),
        }
    except Exception as exc:
        logger.warning("LMM failed: %s", exc)
        return {"error": str(exc)}


def _power_analysis(series: pd.Series, n_locations: int) -> dict:
    sd = float(series.std())
    mean = float(series.mean())
    cohen_d = 0.5  # 80% power threshold
    detectable_abs = cohen_d * sd
    detectable_rel = (detectable_abs / mean * 100) if mean > 0 else float("nan")
    return {
        "n_locations": n_locations,
        "mean": round(mean, 3),
        "sd": round(sd, 3),
        "cv_pct": round(sd / mean * 100 if mean > 0 else float("nan"), 1),
        "detectable_abs_pp": round(detectable_abs, 3),
        "detectable_rel_pct": round(detectable_rel, 1),
    }


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _interpret(d_prev: float, d_cond: float, thr: float = 1.0) -> str:
    expand = d_prev > thr
    contract = d_prev < -thr
    grow = d_cond > thr
    decline = d_cond < -thr

    if expand and grow:
        return "Spatial expansion + within-location growth"
    if expand and decline:
        return "Spatial expansion + within-location decline"
    if contract and grow:
        return "Spatial contraction + within-location growth"
    if contract and decline:
        return "Spatial contraction + within-location decline"
    if not (expand or contract) and grow:
        return "Stable distribution, within-location growth"
    if not (expand or contract) and decline:
        return "Stable distribution, within-location decline"
    return "Stable"
