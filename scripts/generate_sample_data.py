"""
scripts/generate_sample_data.py
--------------------------------
Generate a synthetic CoralNet-format CSV for pipeline testing.

The output matches the real raw export format exactly:
- Short codes as column names (e.g. Acr.arbo, Por.branch) not genus names
- Comma decimal separators (e.g. "23,333") to test the decimal fixer
- Filenames in the standard PBHR pattern (2025.MM_PBHR_TN_meterSIDE.JPG)
- No metadata columns (year, month, etc.) — those are parsed from the filename

This means preprocess.py, growth_forms.py, and the full labelset aggregation
can all be tested without real field data.

Usage
-----
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --out data/raw/sample_coralnet.csv --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ── short codes matching the real PBHR labelset ───────────────────────────────

HARD_CORAL_CODES = {
    # Acropora growth forms
    "Acr.arbo":   4.0,   "Acr.branch": 0.5,   "Acr.caes":  2.0,
    "Acr.corymb": 3.0,   "Acr.digit":  0.5,   "Acr.table": 0.1,
    # Porites growth forms
    "Por.branch": 3.0,   "Por.enc":    0.5,    "Por.mass":  2.0,
    # Isopora growth forms
    "Iso.brug":   2.0,   "Iso.pali":   1.0,
    # Echinopora growth forms
    "Ech.branch": 1.0,   "Ech.encrus": 0.5,   "Ech.folio": 0.5,
    # Montipora growth forms
    "Monti.bran": 1.0,   "Monti.encr": 1.0,   "Monti.foli": 0.5,
    # Galaxea growth forms
    "Galaxea":    1.5,   "Gal.hor":    0.5,
    # Hydnopora growth forms
    "Hyd.micro":  0.5,   "Hyd.rigi":   0.5,
    # Merulina growth forms
    "meru.encru": 0.5,   "meru.folio": 0.5,
    # Single-code genera
    "Anacropora": 0.3,   "Astreo":     0.5,   "Caula":     0.3,
    "Cyphas":     0.5,   "Diploastre": 0.8,   "Dipsast":   0.5,
    "Euphyllia":  0.3,   "Favites":    0.8,   "Fimbria":   0.3,
    "Fungi":      1.5,   "Gar.plan":   0.3,   "Goniastrea": 1.0,
    "Goniopora":  0.5,   "Herpolitha": 0.3,   "Leptoseris": 0.5,
    "Lobophyl":   0.5,   "Mille.bran": 0.3,   "Oxypora":   0.5,
    "Pachyseris": 0.5,   "Pavona":     0.5,   "Pectinia":  0.3,
    "Platygyra":  0.8,   "Pocillo":    0.8,   "Psammo":    0.3,
    "RKC":        0.5,   "Seriatopor": 1.0,   "Stylophora": 0.8,
}

NON_CORAL_CODES = {
    # Algae
    "CCA": 3.0,  "Macroalgae": 2.0,  "Padina": 1.0,  "Turf": 5.0,
    # Hard substrate
    "Dead.coral": 2.0,  "Rock": 8.0,
    # Other invertebrates
    "csponge": 0.5,  "Heliop": 0.5,  "Mille.encr": 0.3,
    "other.inv": 0.5,  "Softy": 1.0,  "Spongetuni": 0.5,
    # Soft substrate
    "Sand": 30.0,
    # Other
    "CorFrame": 1.0,  "Rubble": 5.0,  "TAPE": 0.0,  "Unk": 1.0,
}

MONTHS = [1, 2, 3, 4, 5, 11]
MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 11: "Nov"}
SIDES = ["L", "R"]

ALL_CODES = list(HARD_CORAL_CODES) + list(NON_CORAL_CODES)
HC_WEIGHTS = np.array(list(HARD_CORAL_CODES.values()), dtype=float)
NC_WEIGHTS  = np.array(list(NON_CORAL_CODES.values()),  dtype=float)


def _to_comma_decimal(val: float) -> str:
    """Format as comma-decimal string to match real CoralNet export."""
    return f"{val:.3f}".replace(".", ",")


def generate(
    n_transects: int = 4,
    meters_per_transect: int = 11,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic CoralNet percent-cover CSV.

    Parameters match the real PBHR export:
    - Short codes as column names
    - Comma decimal separators
    - PBHR-style image filenames parseable by preprocess.py
    - Spatial heterogeneity across transects (ICC ~0.7)
    - Slight April temporal effect on hard coral
    """
    rng = np.random.default_rng(seed)
    rows = []

    for transect in range(1, n_transects + 1):
        transect_base = rng.uniform(5, 40)

        for meter in range(1, meters_per_transect + 1):
            location_base = max(0.0, transect_base + rng.normal(0, 8))

            for month in MONTHS:
                temporal_effect = 10 if month == 4 else 0

                for side in SIDES:
                    # Filename in standard PBHR format: 2025.MM_PBHR_TN_meterSIDE.JPG
                    fname = f"2025.{month:02d}_PBHR_T{transect}_{meter}{side}.JPG"
                    image_id = (
                        transect * 100_000 + meter * 1_000 + month * 10
                        + (0 if side == "L" else 1)
                    )

                    row: dict = {
                        "Image ID":          image_id,
                        "Image name":        fname,
                        "Annotation status": "Confirmed",
                        "Points":            50,
                    }

                    # distribute hard coral cover across growth form codes
                    total_hc = max(
                        0.0,
                        location_base + temporal_effect + rng.normal(0, 5)
                    )
                    hc_props = rng.dirichlet(HC_WEIGHTS + 0.1)
                    for code, prop in zip(HARD_CORAL_CODES, hc_props):
                        row[code] = _to_comma_decimal(total_hc * prop)

                    # non-coral
                    remaining = max(0.0, 100.0 - total_hc)
                    nc_props = rng.dirichlet(NC_WEIGHTS + 0.1)
                    for code, prop in zip(NON_CORAL_CODES, nc_props):
                        row[code] = _to_comma_decimal(remaining * prop)

                    rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CoralNet CSV for pipeline testing"
    )
    parser.add_argument(
        "--out", default="data/raw/sample_coralnet.csv",
        help="Output path (default: data/raw/sample_coralnet.csv)",
    )
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--transects", type=int, default=4)
    parser.add_argument("--meters",    type=int, default=11)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = generate(
        n_transects=args.transects,
        meters_per_transect=args.meters,
        seed=args.seed,
    )
    df.to_csv(out, index=False)

    n_images = len(df)
    n_locs   = args.transects * args.meters
    print(f"Saved {n_images} rows ({n_locs} locations × {len(MONTHS)} months × 2 sides)")
    print(f"  Output       : {out}")
    print(f"  Transects    : {args.transects}")
    print(f"  Meters       : {args.meters}")
    print(f"  Months       : {MONTHS}")
    print(f"  Hard coral codes  : {len(HARD_CORAL_CODES)}")
    print(f"  Non-coral codes   : {len(NON_CORAL_CODES)}")
    print()
    print("To run the full pipeline on this data:")
    print("  1. Copy your labelset.csv to data/raw/")
    print("  2. Open notebooks/temporal_analysis.ipynb")
    print(f"  3. Set RAW_PATH = Path('{out}')")


if __name__ == "__main__":
    main()
