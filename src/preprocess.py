"""
preprocess.py
-------------
Transform a raw CoralNet percent-cover export into the two analysis-ready
formats used by this package.

Why preprocessing is a separate step
-------------------------------------
CoralNet exports are annotation-tool outputs.  They contain:

1. **Decimal separator bug** — numbers use a comma instead of a period
   (e.g. ``"23,333"`` instead of ``23.333``).  This is a locale artefact
   from the CoralNet export and must be fixed before any arithmetic.

2. **No survey metadata** — ``year``, ``month``, ``site``, ``transect``,
   ``meter``, and ``side`` are **not** columns in the raw export.  They are
   encoded in the image filename and parsed automatically.  ``lat`` and
   ``lon`` are not in the export at all — add them manually via
   ``build_latlon_template()``.

3. **Fine-grained morphology labels** — e.g. ``Acr.arbo``, ``Acr.branch``,
   ``Acr.caes`` are all *Acropora*.  Two kinds of aggregation are needed:
   - **Functional-group grouping** → ``Hard coral``, ``Algae``, …
   - **Genus-level grouping** → ``Acropora``, ``Porites``, …
   Both are driven by the ``labelset.csv`` lookup table.

4. **Lat/lon** — not present in the raw export.  Add it manually to the
   output CSVs, or use ``build_latlon_template()`` to generate a lookup
   CSV to fill in.

Image filename conventions supported
-------------------------------------
Pattern A — Standard PBHR  (YYYY.MM_SITE_TN_meterSIDE.ext):
    ``2025.01_PBHR_T1_0L.JPG``
    → year=2025, month=1, site=PBHR, transect=1, meter=0, side=L

Pattern B — Variant PBHR  (YYYY.MM.SITE_TN_meterSIDE.ext):
    ``2025.11.PBHR_T2_0L.JPG``
    → year=2025, month=11, site=PBHR, transect=2, meter=0, side=L

Pattern C — OC single-image  (YYYY.MM.DD_SITE_TN_meterM.ext):
    ``2025.04.07_OC_T1_00m.png``
    → year=2025, month=4, site=OC, transect=1, meter=0, side=None

Adding support for a new naming convention
------------------------------------------
Add a dict to ``FILENAME_PATTERNS`` with keys:
    ``pattern``  – compiled regex with named groups:
                   required: year, month, site, transect, meter
                   optional: side
    ``has_side`` – True if the pattern includes a side (L/R) group
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── filename patterns ─────────────────────────────────────────────────────────

FILENAME_PATTERNS: list[dict] = [
    # Pattern A: 2025.01_PBHR_T1_0L.JPG
    {
        "pattern": re.compile(
            r"(?P<year>\d{4})\.(?P<month>\d{2})_(?P<site>[^_]+)"
            r"_T(?P<transect>\d+)_(?P<meter>\d+)(?P<side>[LR])\.\w+",
            re.IGNORECASE,
        ),
        "has_side": True,
    },
    # Pattern B: 2025.11.PBHR_T2_0L.JPG  (dot before site, no leading-zero month)
    {
        "pattern": re.compile(
            r"(?P<year>\d{4})\.(?P<month>\d+)\.(?P<site>[^_]+)"
            r"_T(?P<transect>\d+)_(?P<meter>\d+)(?P<side>[LR])\.\w+",
            re.IGNORECASE,
        ),
        "has_side": True,
    },
    # Pattern C: 2025.04.07_OC_T1_00m.png  (full date, no L/R)
    {
        "pattern": re.compile(
            r"(?P<year>\d{4})\.(?P<month>\d{2})\.\d{2}_(?P<site>[^_]+)"
            r"_T(?P<transect>\d+)_(?P<meter>\d+)m\.\w+",
            re.IGNORECASE,
        ),
        "has_side": False,
    },
]

SIDE_LABELS = {"L": "Left", "R": "Right"}

# Columns that must never be touched by the decimal-separator fixer
_PROTECTED_COLS = {"Image ID", "Image name", "Annotation status"}


# ── public API ────────────────────────────────────────────────────────────────

def preprocess(
    raw_path: str | Path,
    labelset_path: str | Path,
    *,
    sites: Optional[list[str]] = None,
    output_grouped: Optional[str | Path] = None,
    output_genera: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full preprocessing pipeline: raw CoralNet export → two analysis CSVs.

    Parameters
    ----------
    raw_path : str | Path
        Raw CoralNet percent-cover CSV export.
    labelset_path : str | Path
        ``labelset.csv`` — must have columns
        ``Short Code``, ``Functional Group``, ``Genera``.
    sites : list[str], optional
        If given, keep only rows whose parsed site matches one of these
        (case-sensitive).  E.g. ``sites=['PBHR']``.
    output_grouped : str | Path, optional
        If given, write the functional-group CSV to this path.
    output_genera : str | Path, optional
        If given, write the genus-level CSV to this path.

    Returns
    -------
    df_grouped : pd.DataFrame
        One row per image.  Benthic columns = functional groups.
        Lat/lon columns are NOT included — add them manually.
    df_genera : pd.DataFrame
        One row per image.  Benthic columns = genera.
        Lat/lon columns are NOT included — add them manually.

    Notes
    -----
    Lat/lon coordinates are not in the CoralNet export and are excluded
    from both outputs.  Use ``build_latlon_template()`` to generate a
    lookup CSV, fill it in, and join it on ``site × transect × meter``
    before running analyses that need coordinates.
    """
    raw_path = Path(raw_path)
    labelset_path = Path(labelset_path)

    # 1. load
    raw = pd.read_csv(raw_path)
    logger.info("Loaded raw: %d rows × %d columns", *raw.shape)

    # 2. fix decimal separators (comma → period) on all numeric columns
    raw = _fix_decimal_separators(raw)

    # ensure Image ID stays as integer (not coerced to string)
    if "Image ID" in raw.columns:
        raw["Image ID"] = pd.to_numeric(raw["Image ID"], errors="coerce")

    # 3. parse image filenames → metadata columns
    raw = _parse_filenames(raw)

    # 4. drop rows whose filename couldn't be parsed
    before = len(raw)
    raw = raw[raw["year"].notna()].copy()
    skipped = before - len(raw)
    if skipped:
        logger.warning("Dropped %d rows with unparseable filenames", skipped)

    # 5. optional site filter
    if sites:
        raw = raw[raw["site"].isin(sites)].copy()
        logger.info("After site filter %s: %d rows", sites, len(raw))

    # 6. load labelset → build mapping dicts
    labels = pd.read_csv(labelset_path)
    _validate_labelset(labels)

    fg_map: dict[str, str]  = dict(zip(labels["Short Code"], labels["Functional Group"]))
    gen_map: dict[str, str] = dict(zip(labels["Short Code"], labels["Genera"]))

    # 7. identify label value columns (everything that isn't metadata)
    meta_cols = [
        "Image ID", "Image name", "Annotation status", "Points",
        "year", "month", "month_name", "site", "transect", "meter",
        "side_code", "side",
    ]
    value_cols = [c for c in raw.columns if c not in meta_cols]
    unmapped = [c for c in value_cols if c not in fg_map and c not in gen_map]
    if unmapped:
        logger.warning(
            "%d value columns not found in labelset (will be ignored): %s",
            len(unmapped), unmapped,
        )

    # 8. aggregate into functional groups
    fg_groups = sorted(labels["Functional Group"].unique())
    df_grouped = _aggregate(raw, value_cols, fg_map, fg_groups, meta_cols)

    # 9. aggregate into genera
    genera_groups = sorted(labels["Genera"].unique())
    df_genera = _aggregate(raw, value_cols, gen_map, genera_groups, meta_cols)

    # 10. optional write
    if output_grouped:
        Path(output_grouped).parent.mkdir(parents=True, exist_ok=True)
        df_grouped.to_csv(output_grouped, index=False)
        logger.info("Wrote grouped CSV: %s", output_grouped)
    if output_genera:
        Path(output_genera).parent.mkdir(parents=True, exist_ok=True)
        df_genera.to_csv(output_genera, index=False)
        logger.info("Wrote genera CSV: %s", output_genera)

    return df_grouped, df_genera


def build_latlon_template(
    raw_path: str | Path,
    out_path: str | Path = "data/raw/latlon_lookup.csv",
) -> pd.DataFrame:
    """Generate a lat/lon lookup template from the raw export.

    Parses image filenames to find all unique site × transect × meter
    combinations, then writes a CSV with empty ``lat`` / ``lon`` columns
    for you to fill in.  Once filled, the lookup can be joined onto any
    preprocessed dataframe on those three key columns.

    Parameters
    ----------
    raw_path : str | Path
    out_path : str | Path

    Returns
    -------
    pd.DataFrame
    """
    raw = pd.read_csv(raw_path)
    raw = _parse_filenames(raw)
    raw = raw[raw["year"].notna()]

    template = (
        raw[["site", "transect", "meter"]]
        .drop_duplicates()
        .sort_values(["site", "transect", "meter"])
        .assign(lat="", lon="")
        .reset_index(drop=True)
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(out_path, index=False)
    logger.info("Wrote lat/lon template (%d rows) → %s", len(template), out_path)
    return template


# ── private helpers ───────────────────────────────────────────────────────────

def _fix_decimal_separators(df: pd.DataFrame) -> pd.DataFrame:
    """Replace comma-as-decimal-separator in label columns only.

    ``Image ID``, ``Image name``, and ``Annotation status`` are left
    untouched.  Uses numpy arrays to guarantee float64 output, which is
    required because pandas 2.x preserves StringDtype through pd.to_numeric
    assignments and breaks downstream arithmetic.
    """
    import numpy as np

    for col in df.columns:
        if col in _PROTECTED_COLS:
            continue
        s = df[col]
        if s.dtype == object or "str" in str(s.dtype).lower():
            df[col] = np.array(
                [
                    float(str(v).replace(",", ".")) if str(v) not in ("nan", "None", "") else 0.0
                    for v in s
                ],
                dtype=np.float64,
            )
    return df


def _parse_filenames(df: pd.DataFrame) -> pd.DataFrame:
    """Add year, month, month_name, site, transect, meter, side_code, side
    by parsing ``Image name``."""
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }

    parsed: dict[str, list] = {
        "year": [], "month": [], "month_name": [],
        "site": [], "transect": [], "meter": [],
        "side_code": [], "side": [],
    }

    for name in df["Image name"].fillna(""):
        matched = False
        for entry in FILENAME_PATTERNS:
            m = entry["pattern"].match(str(name))
            if m:
                g = m.groupdict()
                yr = int(g["year"])
                mo = int(g["month"])
                parsed["year"].append(yr)
                parsed["month"].append(mo)
                parsed["month_name"].append(month_names.get(mo, str(mo)))
                parsed["site"].append(g["site"].upper())
                parsed["transect"].append(int(g["transect"]))
                parsed["meter"].append(int(g["meter"]))
                side_code = g.get("side", "") if entry["has_side"] else ""
                parsed["side_code"].append(side_code)
                parsed["side"].append(SIDE_LABELS.get(side_code, ""))
                matched = True
                break
        if not matched:
            for key in parsed:
                parsed[key].append(None)
            if name:
                logger.debug("Could not parse filename: %s", name)

    for col, vals in parsed.items():
        df[col] = vals

    df["transect"] = pd.to_numeric(df["transect"], errors="coerce")
    df["meter"]    = pd.to_numeric(df["meter"],    errors="coerce")
    df["year"]     = pd.to_numeric(df["year"],     errors="coerce")
    df["month"]    = pd.to_numeric(df["month"],    errors="coerce")
    return df


def _aggregate(
    df: pd.DataFrame,
    value_cols: list[str],
    mapping: dict[str, str],
    group_names: list[str],
    meta_cols: list[str],
) -> pd.DataFrame:
    """Sum raw label columns into aggregated group columns."""
    present_meta = [c for c in meta_cols if c in df.columns]
    result = df[present_meta].copy()
    for group in group_names:
        codes = [c for c in value_cols if mapping.get(c) == group]
        result[group] = df[codes].sum(axis=1) if codes else 0.0
    return result


def _validate_labelset(labels: pd.DataFrame) -> None:
    required = {"Short Code", "Functional Group", "Genera"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"labelset.csv is missing columns: {missing}")
    if labels["Short Code"].duplicated().any():
        dups = labels[labels["Short Code"].duplicated(keep=False)]["Short Code"].tolist()
        logger.warning("Duplicate Short Codes in labelset (will use first): %s", dups)
