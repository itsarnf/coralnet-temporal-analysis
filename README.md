# CoralNet Temporal Analysis

> **Temporal and growth-form analysis toolkit for [CoralNet](https://coralnet.ucsd.edu/) percent-cover exports.**

This repository contains a reusable Python package (`coralnet_temporal`) and a companion notebook that take a raw CoralNet CSV export and produce:

- Community-level metrics (total coral cover, genera richness, diversity indices)
- Repeated-measures statistical tests (Friedman, Wilcoxon, LMM variance partitioning)
- Genus-level temporal analysis with prevalence / conditional mean decomposition
- Growth-form analysis for any genus with multiple morphological codes (Porites, Acropora, and 6 others)
- Bray-Curtis compositional dissimilarity
- Publication-ready figures

Developed for **PBHR reef monitoring** at Seachange Indonesia (January–November 2025, 44 permanent locations, 6 time points, 33 coral genera). Works with any CoralNet export that follows the standard filename convention.

---

## Workflow overview

```
raw CoralNet CSV  +  labelset.csv
        │
        ▼  preprocess()
        │  • fixes comma decimal separators
        │  • parses year / month / site / transect / meter / side from filename
        │  • aggregates fine morphology codes → functional groups & genera
        │  • supports 3 filename patterns (PBHR standard, PBHR variant, OC)
        ▼
percent_covers_grouped.csv        percent_covers_genera.csv
(Hard coral, Algae, …)            (Acropora, Porites, …)
        │                                    │
        └──────────────┬─────────────────────┘
                       ▼
              load_coralnet_export()
              compute_community_metrics()
              run_temporal_tests()          ← Friedman, Wilcoxon, LMM, power
              run_genera_analysis()         ← genus-level Friedman + prevalence
              prepare_growth_form_data()    ← branching vs massive vs encrusting
              run_growth_form_tests()
                       │
                       ▼
              statistics + figures
              (add lat/lon manually to CSVs for spatial work)
```

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/itsarnf/coralnet-temporal-analysis.git
cd coralnet-temporal-analysis

# 2. Install with LMM support (recommended)
pip install -e ".[lmm]"

# 3. Verify — should print: 48 passed
python -m pytest tests/ -q

# 4. Generate synthetic data and run the notebook
python scripts/generate_sample_data.py
jupyter lab notebooks/temporal_analysis.ipynb
```

---

## Package structure

```
src/coralnet_temporal/
├── __init__.py       — public API (v1.2.0)
├── preprocess.py     — raw CoralNet CSV → grouped CSV + genera CSV
├── loader.py         — load + validate preprocessed CSVs
├── community.py      — cover, richness, Shannon, Simpson, Evenness
├── temporal.py       — Friedman, Wilcoxon, LMM, power, Bray-Curtis
├── genera.py         — genus-level temporal tests & prevalence analysis
├── growth_forms.py   — growth-form analysis for any genus
└── visualize.py      — all plot functions (community, genera, growth forms)
```

---

## Compatibility with CoralNet exports

CoralNet produces one CSV row per annotated image. The raw export is used directly — no manual formatting required.

**Required metadata columns** (parsed from the image filename, not present as columns in the raw export):

| Field | Source | Example |
|-------|--------|---------|
| `year` | filename | `2025` |
| `month` | filename | `1` (January) |
| `site` | filename | `PBHR` |
| `transect` | filename | `1` |
| `meter` | filename | `5` |
| `side` | filename | `L` or `R` |

**Supported filename patterns:**

| Pattern | Example | Notes |
|---------|---------|-------|
| Standard PBHR | `2025.01_PBHR_T1_5L.JPG` | Most survey months |
| Variant PBHR | `2025.11.PBHR_T2_0L.JPG` | Dot before site name |
| OC single-image | `2025.04.07_OC_T1_05m.png` | No L/R, full date |

**Benthic labels** are aggregated using `labelset.csv` — one row per CoralNet short code mapping it to a functional group and a genus. Non-coral categories (Dead coral, Sand, Rubble, etc.) are excluded from coral analyses automatically.

**Lat/lon** is not in the CoralNet export. Use `build_latlon_template()` to generate a lookup CSV to fill in manually, then join it on `site × transect × meter`.

---

## Statistical design

| Design choice | Reason |
|---|---|
| Average left + right images before analysis | One independent observation per location × time — prevents pseudoreplication |
| Friedman test (not repeated-measures ANOVA) | Coral cover data are non-normal and heteroscedastic |
| Wilcoxon signed-rank for pairwise comparisons | Paired, non-parametric equivalent of a paired t-test |
| LMM with Location as random effect | Quantifies spatial vs temporal variance (ICC) |
| No transect-level statistical inference | Only 4 transects — no spatial replication |
| Power analysis with Cohen's d = 0.5 | Conservative 80%-power threshold |

### Key findings (PBHR 2025)
- **ICC = 0.86** — spatial heterogeneity dominates temporal variation
- Minimum detectable change at 80% power: ~±9.3 pp (~±60%)
- April spike in hard coral cover (23.8%) is statistically significant (Wilcoxon p < 0.001 for Mar→Apr and Apr→May)
- Bray-Curtis Jan–Nov = 0.23 → minor compositional change
- Porites Massive shows significant temporal variation (Friedman p = 0.0001); Branching does not

---

## Usage

```python
from coralnet_temporal import (
    preprocess,
    build_latlon_template,
    load_coralnet_export,
    compute_community_metrics,
    run_temporal_tests,
    run_genera_analysis,
    prepare_growth_form_data,
    growth_form_stats,
    run_growth_form_tests,
    growth_form_ratio,
    GROWTH_FORM_PRESETS,
)
from coralnet_temporal.temporal import bray_curtis_dissimilarity
from coralnet_temporal.genera import compute_genera_stats
from coralnet_temporal.visualize import (
    plot_community_temporal,
    plot_growth_form_temporal,
)

# 1. Preprocess raw CoralNet export
df_grouped, df_genera = preprocess(
    "data/raw/2025_percent_covers.csv",
    "data/raw/labelset.csv",
    sites=["PBHR"],
    output_grouped="data/processed/grouped.csv",
    output_genera="data/processed/genera.csv",
)

# 2. Load preprocessed data for analysis
df, coral_cols, _ = load_coralnet_export("data/processed/grouped.csv")

# 3. Community metrics
df = compute_community_metrics(df, coral_cols)

# 4. Temporal tests
results = run_temporal_tests(df, metric="Total_Coral")
print(results["friedman"])  # {'statistic': 51.02, 'p_value': 0.0, 'significant': True, 'n': 44}
print(results["lmm"])       # {'icc': 0.858, 'interpretation': 'Spatial heterogeneity dominates...'}

# 5. Genus-level analysis
temporal_df, prevalence_df = run_genera_analysis(df, coral_cols)

# 6. Bray-Curtis dissimilarity
bc = bray_curtis_dissimilarity(df, coral_cols, "Jan", "Nov")

# 7. Growth form analysis — change genus to any key in GROWTH_FORM_PRESETS
df_gf = prepare_growth_form_data(
    "data/raw/2025_percent_covers.csv",
    form_map=GROWTH_FORM_PRESETS["Porites"],   # or "Acropora", "Echinopora", …
    sites=["PBHR"],
    genus_name="Porites",
)
stats  = growth_form_stats(df_gf, list(GROWTH_FORM_PRESETS["Porites"]))
tests  = run_growth_form_tests(df_gf, list(GROWTH_FORM_PRESETS["Porites"]))
ratios = growth_form_ratio(df_gf, "Branching", "Massive", by="transect")

# 8. Figures
fig = plot_community_temporal(df, month_order=["Jan","Feb","Mar","Apr","May","Nov"])
fig.savefig("output.png", dpi=150)

fig = plot_growth_form_temporal(
    df_gf, list(GROWTH_FORM_PRESETS["Porites"]),
    genus_name="Porites",
    month_order=["Jan","Feb","Mar","Apr","May","Nov"],
    ratio_pair=("Branching", "Massive"),
)
fig.savefig("porites_temporal.png", dpi=150)
```

### Built-in growth form presets

| Genus | Forms |
|-------|-------|
| Porites | Branching, Encrusting, Massive |
| Acropora | Arborescent, Branching, Caespitose, Corymbose, Digitate, Tabulate |
| Echinopora | Branching, Encrusting, Foliose |
| Montipora | Branching, Encrusting, Foliose |
| Hydnopora | Massive, Branching |
| Merulina | Encrusting, Foliose |

For genera not in the presets, pass a custom `form_map` dict with your own short codes.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest            # 48 tests, ~15s
pytest -q         # quiet output
```

---

## Data

Raw CSV files are **not committed** to this repository (field data). Place your CoralNet export and labelset at:

```
data/raw/2025_percent_covers.csv   ← raw CoralNet export
data/raw/labelset.csv              ← short code → functional group + genus
```

Then update `RAW_PATH` and `LABELSET_PATH` in the notebook configuration cell.

To test the pipeline without real data:

```bash
python scripts/generate_sample_data.py
# writes data/raw/sample_coralnet.csv
```

---

## Contributing

1. Fork → feature branch → pull request
2. Run `black src/ tests/` and `ruff src/ tests/` before committing
3. Add or update tests for any new functionality — aim to keep coverage above 80%

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

If you use this code in a publication, please cite:

> Fuadi, I. (2026). *CoralNet Temporal Analysis Toolkit* (v1.2.0). Seachange Indonesia. GitHub. https://github.com/itsarnf/coralnet-temporal-analysis
