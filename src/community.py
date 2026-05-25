"""
community.py
------------
Compute community-level benthic metrics from a prepared CoralNet dataframe.

Metrics
-------
Total_Coral     sum of all coral category cover at each location × time
Richness        number of genera / morphologies with cover > 0
Shannon_H       Shannon diversity index (base e)
Simpson_D       Simpson's dominance index (1 - D form)
Evenness        Pielou's J (Shannon / log(Richness))
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── public API ────────────────────────────────────────────────────────────────

def compute_community_metrics(
    df: pd.DataFrame,
    coral_cols: list[str],
) -> pd.DataFrame:
    """Add community-level summary columns to *df* in place and return it.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`~coralnet_temporal.loader.load_coralnet_export`.
    coral_cols : list[str]
        Column names that represent hard-coral genera / morphologies.

    Returns
    -------
    pd.DataFrame
        Same dataframe with additional columns:
        ``Total_Coral``, ``Richness``, ``Shannon_H``,
        ``Simpson_D``, ``Evenness``.
    """
    cover = df[coral_cols].clip(lower=0)

    df["Total_Coral"] = cover.sum(axis=1)
    df["Richness"] = (cover > 0).sum(axis=1)
    df["Shannon_H"] = cover.apply(_shannon, axis=1)
    df["Simpson_D"] = cover.apply(_simpson, axis=1)
    df["Evenness"] = df.apply(
        lambda r: r["Shannon_H"] / np.log(r["Richness"]) if r["Richness"] > 1 else 0.0,
        axis=1,
    )
    return df


def monthly_summary(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    month_order: list[str] | None = None,
) -> pd.DataFrame:
    """Return mean ± SD of community metrics grouped by month.

    Parameters
    ----------
    df : pd.DataFrame
        Must have already been processed by :func:`compute_community_metrics`.
    metrics : list[str], optional
        Which metric columns to summarise.  Defaults to
        ``['Total_Coral', 'Richness', 'Shannon_H']``.
    month_order : list[str], optional
        Ordered list of month-name strings for sorting.

    Returns
    -------
    pd.DataFrame
        Multi-level column dataframe with mean / std / median per metric.
    """
    if metrics is None:
        metrics = ["Total_Coral", "Richness", "Shannon_H"]

    summary = (
        df.groupby("month_name")[metrics]
        .agg(["mean", "std", "median"])
    )
    summary.columns = ["_".join(c) for c in summary.columns]

    if month_order:
        summary = summary.reindex(
            [m for m in month_order if m in summary.index]
        )
    return summary.round(3)


# ── private helpers ───────────────────────────────────────────────────────────

def _shannon(row: pd.Series) -> float:
    p = row[row > 0]
    if p.empty:
        return 0.0
    total = p.sum()
    if total == 0:
        return 0.0
    p = p / total
    return float(-(p * np.log(p)).sum())


def _simpson(row: pd.Series) -> float:
    p = row[row > 0]
    if p.empty:
        return 0.0
    total = p.sum()
    if total == 0:
        return 0.0
    p = p / total
    return float(1 - (p ** 2).sum())
