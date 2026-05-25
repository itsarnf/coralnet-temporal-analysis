"""
genera.py
---------
Genus-level temporal analysis for CoralNet percent-cover data.

Functions
---------
compute_genera_stats          Mean cover, max cover, prevalence for all genera
run_genera_analysis           Friedman test + prevalence/conditional analysis
                              for each common genus
filter_common_genera          Subset to genera meeting minimum criteria
"""

from __future__ import annotations

import logging

import pandas as pd
from scipy import stats

from .temporal import _sig_stars, _interpret

logger = logging.getLogger(__name__)


# ── public API ────────────────────────────────────────────────────────────────

def compute_genera_stats(
    df: pd.DataFrame,
    coral_cols: list[str],
) -> pd.DataFrame:
    """Compute summary statistics for every coral genus / morphology.

    Parameters
    ----------
    df : pd.DataFrame
    coral_cols : list[str]

    Returns
    -------
    pd.DataFrame
        Index = genus name.  Columns: ``Mean_Cover``, ``Max_Cover``,
        ``Prevalence``.  Sorted by mean cover descending.
    """
    return pd.DataFrame(
        {
            "Mean_Cover (%)": df[coral_cols].mean(),
            "Max_Cover (%)": df[coral_cols].max(),
            "Prevalence (%)": (df[coral_cols] > 0).mean() * 100,
        }
    ).sort_values("Mean_Cover (%)", ascending=False)


def filter_common_genera(
    genera_stats: pd.DataFrame,
    min_mean_cover: float = 0.1,
    min_prevalence: float = 5.0,
) -> list[str]:
    """Return genera that meet at least one of the two inclusion criteria.

    Parameters
    ----------
    genera_stats : pd.DataFrame
        Output of :func:`compute_genera_stats`.
    min_mean_cover : float
        Minimum mean percent cover threshold.
    min_prevalence : float
        Minimum prevalence (% of observations) threshold.

    Returns
    -------
    list[str]
        Genus names meeting criteria, ordered by mean cover.
    """
    mask = (
        (genera_stats["Mean_Cover (%)"] > min_mean_cover)
        | (genera_stats["Prevalence (%)"] > min_prevalence)
    )
    return genera_stats.index[mask].tolist()


def run_genera_analysis(
    df: pd.DataFrame,
    coral_cols: list[str],
    *,
    min_mean_cover: float = 0.1,
    min_prevalence: float = 5.0,
    min_sites: int = 5,
    month_a: str = "Jan",
    month_b: str = "Nov",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Friedman tests and prevalence/conditional analysis for all common
    coral genera.

    Parameters
    ----------
    df : pd.DataFrame
    coral_cols : list[str]
    min_mean_cover : float
        Genera with lower mean cover are skipped (unless prevalence qualifies).
    min_prevalence : float
        Genera seen at fewer locations are skipped (unless cover qualifies).
    min_sites : int
        Minimum number of locations where the genus is present for a Friedman
        test to be run.
    month_a, month_b : str
        First and last month names for the prevalence / conditional comparison.

    Returns
    -------
    temporal_df : pd.DataFrame
        Friedman test results sorted by p-value.
    prevalence_df : pd.DataFrame
        Prevalence vs conditional mean table.
    """
    genera_stats = compute_genera_stats(df, coral_cols)
    common = filter_common_genera(genera_stats, min_mean_cover, min_prevalence)
    logger.info("Running genus-level tests for %d common genera", len(common))

    temporal_rows = []
    for genus in common:
        wide = (
            df.pivot(index="location_id", columns="month", values=genus)
            .dropna()
        )
        # restrict to locations where the genus is present at least once
        present = wide[wide.sum(axis=1) > 0]
        if len(present) < min_sites:
            logger.debug("Skipping %s: only %d sites with presence", genus, len(present))
            continue
        try:
            stat, p = stats.friedmanchisquare(
                *[present[col].values for col in present.columns]
            )
        except ValueError as exc:
            logger.warning("Friedman failed for %s: %s", genus, exc)
            continue

        first_val = present.iloc[:, 0].mean()
        last_val = present.iloc[:, -1].mean()
        change = last_val - first_val
        pct_change = (change / (first_val + 0.01)) * 100

        temporal_rows.append(
            {
                "Genus": genus,
                "n_sites": len(present),
                "Mean_Cover (%)": round(float(genera_stats.loc[genus, "Mean_Cover (%)"]), 3),
                "Prevalence (%)": round(float(genera_stats.loc[genus, "Prevalence (%)"]), 1),
                f"{month_a} (%)": round(first_val, 3),
                f"{month_b} (%)": round(last_val, 3),
                "Change (pp)": round(change, 3),
                "Rel_Change (%)": round(pct_change, 1),
                "Chi2": round(float(stat), 3),
                "p_value": round(float(p), 6),
                "sig": _sig_stars(p),
            }
        )

    temporal_df = (
        pd.DataFrame(temporal_rows)
        .sort_values("p_value")
        .reset_index(drop=True)
    )

    # prevalence / conditional analysis for the same genera
    from .temporal import prevalence_conditional_analysis  # avoid circular import
    prevalence_df = prevalence_conditional_analysis(df, common, month_a, month_b)

    return temporal_df, prevalence_df
