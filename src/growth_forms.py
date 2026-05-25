"""
growth_forms.py
---------------
Growth-form-level analysis for coral genera with multiple morphological
short codes in the labelset.

What are growth forms?
----------------------
CoralNet labels distinguish morphological variants within a genus.
For example, Porites is recorded as three separate codes:

    Por.branch  →  Branching Porites
    Por.enc     →  Encrusting Porites
    Por.mass    →  Massive Porites

The genus-level pipeline (genera.py) sums these into a single ``Porites``
column, which is correct for community analysis.  But when a genus is
ecologically important — or when its growth-form ratio is itself a
monitoring metric — you want to split them back out and analyse each
form separately.

This module works from the **raw CoralNet CSV** (before label aggregation),
not from the preprocessed genera CSV.

Genera with multiple growth-form codes in PBHR labelset
--------------------------------------------------------
Genus           Forms
Acropora        Arborescent, Branching, Caespitose, Corymbose, Digitate, Tabulate
Porites         Branching, Encrusting, Massive
Echinopora      Branching, Encrusting, Foliose
Montipora       Branching, Encrusting, Foliose
Hydnopora       Massive, Branching
Merulina        Encrusting, Foliose

Configuring growth forms
------------------------
Pass a ``form_map`` dict to any function:

    form_map = {
        "Arborescent": ["Acr.arbo"],
        "Branching":   ["Acr.branch"],
        "Caespitose":  ["Acr.caes"],
        "Corymbose":   ["Acr.corymb"],
        "Digitate":    ["Acr.digit"],
        "Tabulate":    ["Acr.table"],
    }

Each key is the display name; the value is the list of raw short codes
that map to it (usually one, but can be multiple).

Built-in presets are available via ``GROWTH_FORM_PRESETS``.

Typical usage
-------------
>>> from coralnet_temporal.growth_forms import (
...     prepare_growth_form_data,
...     growth_form_stats,
...     run_growth_form_tests,
...     GROWTH_FORM_PRESETS,
... )
>>> df_gf = prepare_growth_form_data(
...     raw_path="data/raw/2025_percent_covers.csv",
...     form_map=GROWTH_FORM_PRESETS["Acropora"],
...     sites=["PBHR"],
... )
>>> stats_df  = growth_form_stats(df_gf, list(GROWTH_FORM_PRESETS["Acropora"]))
>>> tests_df  = run_growth_form_tests(df_gf, list(GROWTH_FORM_PRESETS["Acropora"]))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .preprocess import _fix_decimal_separators, _parse_filenames
from .temporal import _sig_stars

logger = logging.getLogger(__name__)

# ── built-in presets ──────────────────────────────────────────────────────────

GROWTH_FORM_PRESETS: dict[str, dict[str, list[str]]] = {
    "Porites": {
        "Branching":  ["Por.branch"],
        "Encrusting": ["Por.enc"],
        "Massive":    ["Por.mass"],
    },
    "Acropora": {
        "Arborescent": ["Acr.arbo"],
        "Branching":   ["Acr.branch"],
        "Caespitose":  ["Acr.caes"],
        "Corymbose":   ["Acr.corymb"],
        "Digitate":    ["Acr.digit"],
        "Tabulate":    ["Acr.table"],
    },
    "Echinopora": {
        "Branching":   ["Ech.branch"],
        "Encrusting":  ["Ech.encrus"],
        "Foliose":     ["Ech.folio"],
    },
    "Montipora": {
        "Branching":   ["Monti.bran"],
        "Encrusting":  ["Monti.encr"],
        "Foliose":     ["Monti.foli"],
    },
    "Hydnopora": {
        "Massive":     ["Hyd.micro"],
        "Branching":   ["Hyd.rigi"],
    },
    "Merulina": {
        "Encrusting":  ["meru.encru"],
        "Foliose":     ["meru.folio"],
    },
}

_MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

# ── public API ────────────────────────────────────────────────────────────────

def prepare_growth_form_data(
    raw_path: str | Path,
    form_map: dict[str, list[str]],
    *,
    sites: Optional[list[str]] = None,
    genus_name: str = "Genus",
) -> pd.DataFrame:
    """Load the raw CoralNet CSV and return a location × time dataframe with
    one column per growth form, plus a ``{genus_name}_Total`` column.

    Parameters
    ----------
    raw_path : str | Path
        Raw CoralNet percent-cover CSV (before any preprocessing).
    form_map : dict[str, list[str]]
        ``{"Form display name": ["short_code_1", ...], ...}``
        Use ``GROWTH_FORM_PRESETS["Porites"]`` as a starting point.
    sites : list[str], optional
        If given, keep only rows whose parsed site is in this list.
    genus_name : str
        Used to name the ``{genus_name}_Total`` column.  Defaults to the
        first key's parent genus if you pass a preset, otherwise pass it
        explicitly (e.g. ``genus_name="Acropora"``).

    Returns
    -------
    pd.DataFrame
        Rows: one per location × time (left/right already averaged).
        Columns: metadata + one column per form name + ``{genus_name}_Total``.
    """
    raw_path = Path(raw_path)
    raw = pd.read_csv(raw_path)

    # fix decimals and parse filenames (same logic as preprocess.py)
    raw = _fix_decimal_separators(raw)
    raw = _parse_filenames(raw)

    # drop unparseable rows
    raw = raw[raw["year"].notna()].copy()

    if sites:
        raw = raw[raw["site"].isin(sites)].copy()

    # collect all short codes referenced by this form_map
    all_codes = [code for codes in form_map.values() for code in codes]
    missing = [c for c in all_codes if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Short codes not found in {raw_path.name}: {missing}\n"
            "Check that these codes exist in the raw CoralNet export."
        )

    # average left/right images per location × time
    meta_group = ["month", "month_name", "transect", "meter"]
    df = raw.groupby(meta_group)[all_codes].mean().reset_index()
    df["location_id"] = df["transect"].astype(str) + "_" + df["meter"].astype(str)

    # build one column per form (sum its short codes)
    form_cols = list(form_map.keys())
    for form, codes in form_map.items():
        df[form] = df[[c for c in codes if c in df.columns]].sum(axis=1)

    # total column
    df[f"{genus_name}_Total"] = df[form_cols].sum(axis=1)

    # drop the raw short-code columns — keep only the named forms
    df = df.drop(columns=all_codes, errors="ignore")

    logger.info(
        "Growth form data prepared: %d observations, %d locations, "
        "%d time points, %d forms",
        len(df),
        df["location_id"].nunique(),
        df["month"].nunique(),
        len(form_cols),
    )
    return df


def growth_form_stats(
    df: pd.DataFrame,
    form_cols: list[str],
    genus_total_col: str | None = None,
    month_order: list[str] | None = None,
) -> dict:
    """Compute summary statistics for each growth form.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`prepare_growth_form_data`.
    form_cols : list[str]
        Growth form column names.
    genus_total_col : str, optional
        Name of the total column (e.g. ``"Acropora_Total"``).
        If omitted, total is computed on-the-fly from ``form_cols``.
    month_order : list[str], optional
        Display order for month-level summaries.

    Returns
    -------
    dict with keys
        ``overall`` — pd.DataFrame: mean/max/prevalence per form
        ``by_transect`` — pd.DataFrame: mean cover per transect
        ``by_month`` — pd.DataFrame: mean cover per month
        ``proportions_overall`` — pd.Series: % share of each form
        ``proportions_by_transect`` — pd.DataFrame
        ``proportions_by_month`` — pd.DataFrame
        ``jan_nov_change`` — pd.DataFrame (if both Jan and Nov present)
        ``ratio_table`` — pd.DataFrame: pairwise ratios if exactly 2 forms
    """
    if genus_total_col is None:
        genus_total_col = "_Total"
        df = df.copy()
        df["_Total"] = df[form_cols].sum(axis=1)

    # overall
    totals = df[form_cols].sum()
    overall = pd.DataFrame({
        "Mean Cover (%)": df[form_cols].mean(),
        "Max Cover (%)":  df[form_cols].max(),
        "Prevalence (%)": (df[form_cols] > 0).mean() * 100,
        "Total Cover":    totals,
        "Share (%)":      (totals / totals.sum() * 100).round(1),
    })

    # by transect
    by_transect = df.groupby("transect")[form_cols + [genus_total_col]].mean().round(3)
    transect_totals = df.groupby("transect")[form_cols].sum()
    props_by_transect = transect_totals.div(transect_totals.sum(axis=1), axis=0) * 100

    # by month
    mn = df.groupby("month_name")[form_cols + [genus_total_col]].mean()
    if month_order:
        mn = mn.reindex([m for m in month_order if m in mn.index])
    by_month = mn.round(3)

    month_totals = df.groupby("month_name")[form_cols].sum()
    if month_order:
        month_totals = month_totals.reindex([m for m in month_order if m in month_totals.index])
    props_by_month = month_totals.div(month_totals.sum(axis=1), axis=0) * 100

    result = {
        "overall":                overall,
        "by_transect":            by_transect,
        "by_month":               by_month,
        "proportions_overall":    overall["Share (%)"],
        "proportions_by_transect": props_by_transect,
        "proportions_by_month":   props_by_month,
    }

    # Jan → Nov change table
    if "Jan" in by_month.index and "Nov" in by_month.index:
        jan = by_month.loc["Jan", form_cols]
        nov = by_month.loc["Nov", form_cols]
        delta = nov - jan
        pct_delta = (delta / jan.replace(0, np.nan) * 100).round(1)
        result["jan_nov_change"] = pd.DataFrame({
            "Jan (%)":  jan.round(3),
            "Nov (%)":  nov.round(3),
            "Δ (pp)":   delta.round(3),
            "Δ (%)":    pct_delta,
        })

    # branch:massive (or any 2-form) ratio by transect
    if len(form_cols) == 2:
        f1, f2 = form_cols
        ratio_name = f"{f1}:{f2} ratio"
        r_transect = (
            by_transect[f1] / by_transect[f2].replace(0, np.nan)
        ).round(2).rename(ratio_name)
        result["ratio_table"] = r_transect.to_frame()

    return result


def run_growth_form_tests(
    df: pd.DataFrame,
    form_cols: list[str],
    *,
    min_sites: int = 5,
) -> pd.DataFrame:
    """Run Friedman repeated-measures tests for each growth form.

    Only locations where the form is present at least once are included,
    and only forms present at ``min_sites`` or more locations are tested.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`prepare_growth_form_data`.
    form_cols : list[str]
        Growth form column names.
    min_sites : int
        Minimum number of locations with presence for the test to run.

    Returns
    -------
    pd.DataFrame
        Columns: Form, n_sites, Chi2, p_value, sig, plus directional note.
    """
    rows = []
    for form in form_cols:
        wide = (
            df.pivot(index="location_id", columns="month", values=form)
            .dropna()
        )
        present = wide[wide.sum(axis=1) > 0]
        if len(present) < min_sites:
            logger.debug("Skipping %s: only %d sites with presence", form, len(present))
            rows.append({
                "Form": form, "n_sites": len(present),
                "Chi2": np.nan, "p_value": np.nan, "sig": "—",
                "Note": f"skipped (n={len(present)} < {min_sites})",
            })
            continue
        try:
            stat, p = stats.friedmanchisquare(
                *[present[col].values for col in present.columns]
            )
        except ValueError as exc:
            logger.warning("Friedman failed for %s: %s", form, exc)
            continue

        # directional summary: first vs last month
        first_mean = present.iloc[:, 0].mean()
        last_mean  = present.iloc[:, -1].mean()
        direction  = "↑" if last_mean > first_mean else ("↓" if last_mean < first_mean else "→")

        rows.append({
            "Form":    form,
            "n_sites": len(present),
            "Chi2":    round(float(stat), 3),
            "p_value": round(float(p), 6),
            "sig":     _sig_stars(p),
            "Note":    f"first={first_mean:.3f}% → last={last_mean:.3f}% {direction}",
        })

    return pd.DataFrame(rows)


def growth_form_ratio(
    df: pd.DataFrame,
    numerator: str,
    denominator: str,
    *,
    by: str = "overall",
    month_order: list[str] | None = None,
) -> pd.Series:
    """Compute the ratio of two growth forms, optionally grouped.

    Parameters
    ----------
    df : pd.DataFrame
    numerator, denominator : str
        Form column names.
    by : str
        ``"overall"``, ``"transect"``, or ``"month"``.
    month_order : list[str], optional

    Returns
    -------
    pd.Series
    """
    if by == "overall":
        n = df[numerator].mean()
        d = df[denominator].mean()
        return pd.Series(
            {f"{numerator}:{denominator}": n / d if d > 0 else np.nan}
        )
    elif by == "transect":
        means = df.groupby("transect")[[numerator, denominator]].mean()
        return (means[numerator] / means[denominator].replace(0, np.nan)).round(2)
    elif by == "month":
        means = df.groupby("month_name")[[numerator, denominator]].mean()
        if month_order:
            means = means.reindex([m for m in month_order if m in means.index])
        return (means[numerator] / means[denominator].replace(0, np.nan)).round(2)
    else:
        raise ValueError(f"by must be 'overall', 'transect', or 'month', got '{by}'")
