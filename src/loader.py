"""
loader.py
---------
Load and validate CoralNet percent-cover CSV exports for temporal analysis.

CoralNet exports one row per image.  Each monitoring location is photographed
from two sides (left / right), so the first step is to average those replicates
to obtain one value per location × time combination — the true independent unit.

Expected input columns (flexible; extras are kept as metadata)
--------------------------------------------------------------
Required
    year, month, transect, meter

Optional but used when present
    site, side / side_code, lat, lon, Image ID / Image name,
    Annotation status, Points

Everything else is treated as a benthic category.

Non-coral categories that are excluded from coral analyses by default
---------------------------------------------------------------------
Dead coral, Sponge, Heliopora, Millepora, Other sessile, Soft coral,
Sand, Rock, Other, Rubble, Coralline algae, Macroalgae, Turf algae
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── defaults ─────────────────────────────────────────────────────────────────

DEFAULT_NON_CORAL: list[str] = [
    "Dead coral", "Sponge", "Heliopora", "Millepora",
    "Other sessile", "Soft coral", "Sand", "Rock",
    "Other", "Rubble", "Coralline algae", "Macroalgae", "Turf algae",
]

METADATA_COLS: list[str] = [
    "Image ID", "Image name", "year", "month", "month_name", "site",
    "transect", "meter", "location_id", "side_code", "side", "lat", "lon",
    "Annotation status", "Points",
]

MONTH_NAMES: dict[int, str] = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


# ── public API ────────────────────────────────────────────────────────────────

def load_coralnet_export(
    file_path: str | Path,
    *,
    non_coral: Optional[list[str]] = None,
    extra_exclude: Optional[list[str]] = None,
    average_sides: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load a CoralNet percent-cover CSV export and prepare it for analysis.

    Parameters
    ----------
    file_path : str or Path
        Path to the CoralNet CSV export.
    non_coral : list[str], optional
        Full list of non-coral category labels to exclude from the coral
        column set.  Defaults to ``DEFAULT_NON_CORAL``.
    extra_exclude : list[str], optional
        Additional category names to exclude on top of ``non_coral``.
    average_sides : bool
        If ``True`` (default), average left and right images so that each
        location × time cell is one independent observation.

    Returns
    -------
    df : pd.DataFrame
        Cleaned, averaged data ready for analysis.
    coral_cols : list[str]
        Column names that represent hard-coral genera / morphologies.
    all_benthic_cols : list[str]
        All benthic category columns (coral + non-coral) present in the file.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    raw = pd.read_csv(file_path)
    logger.info("Loaded %d rows × %d columns from %s", *raw.shape, file_path.name)

    # ── resolve exclusion lists ───────────────────────────────────────────────
    exclude = set(non_coral if non_coral is not None else DEFAULT_NON_CORAL)
    if extra_exclude:
        exclude.update(extra_exclude)

    # ── identify benthic category columns ────────────────────────────────────
    all_benthic_cols = [
        c for c in raw.columns
        if c not in METADATA_COLS
    ]
    coral_cols = [c for c in all_benthic_cols if c not in exclude]

    if not coral_cols:
        raise ValueError(
            "No coral columns found after exclusions. "
            "Check that the CSV contains genus/morphology columns."
        )

    # ── coerce to numeric ─────────────────────────────────────────────────────
    for col in all_benthic_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0)

    # ── add helper columns ────────────────────────────────────────────────────
    raw["month_name"] = raw["month"].map(MONTH_NAMES)
    raw["transect"] = raw["transect"].astype(str)
    raw["meter"] = raw["meter"].astype(str)
    raw["location_id"] = raw["transect"] + "_" + raw["meter"]

    # ── average left / right ──────────────────────────────────────────────────
    group_cols = ["year", "month", "month_name", "transect", "meter", "location_id"]
    # include optional metadata if present
    for opt in ("site", "lat", "lon"):
        if opt in raw.columns:
            group_cols.append(opt)

    if average_sides:
        df = (
            raw.groupby(group_cols, sort=False)[all_benthic_cols]
            .mean()
            .reset_index()
        )
        logger.info(
            "After averaging sides: %d location × time observations, "
            "%d unique locations, %d time points",
            len(df),
            df["location_id"].nunique(),
            df["month"].nunique(),
        )
    else:
        df = raw.copy()

    _validate(df, coral_cols)
    return df, coral_cols, all_benthic_cols


# ── helpers ───────────────────────────────────────────────────────────────────

def _validate(df: pd.DataFrame, coral_cols: list[str]) -> None:
    """Basic sanity checks on the loaded dataframe."""
    required = {"month", "transect", "meter", "location_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing after loading: {missing}")

    n_loc = df["location_id"].nunique()
    n_time = df["month"].nunique()
    logger.info("Validation passed: %d locations × %d time points", n_loc, n_time)

    if n_loc < 2:
        logger.warning("Only %d unique location(s) — check data.", n_loc)
    if n_time < 2:
        logger.warning("Only %d time point(s) — temporal tests require ≥2.", n_time)
    if len(coral_cols) == 0:
        raise ValueError("No coral category columns identified.")
