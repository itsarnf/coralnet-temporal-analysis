"""
coralnet_temporal
=================
Temporal and growth-form analysis toolkit for CoralNet percent-cover exports.

Modules
-------
preprocess   – Raw CoralNet export → analysis-ready CSVs
              (fixes decimal separators, parses filenames, aggregates labels)
loader       – Load and validate already-preprocessed CoralNet CSVs
community    – Community-level metrics (cover, richness, diversity)
temporal     – Repeated-measures statistical tests
genera       – Genus-level temporal analysis
growth_forms – Growth-form analysis for any genus with multiple morphologies
visualize    – Publication-ready plots (community, genera, growth forms)

Typical workflow
----------------
1. preprocess()               raw CoralNet CSV → grouped CSV + genera CSV
2. load_coralnet_export()     grouped or genera CSV → analysis dataframe
3. compute_community_metrics() → adds cover/richness/diversity columns
4. run_temporal_tests()       → Friedman, Wilcoxon, LMM, power analysis
5. run_genera_analysis()      → genus-level temporal tests
6. prepare_growth_form_data() → per-form data for a chosen genus
   run_growth_form_tests()    → Friedman test per growth form
   growth_form_ratio()        → branching:massive or any form ratio
"""

from .preprocess import preprocess, build_latlon_template
from .loader import load_coralnet_export
from .community import compute_community_metrics
from .temporal import run_temporal_tests
from .genera import run_genera_analysis
from .growth_forms import (
    prepare_growth_form_data,
    growth_form_stats,
    run_growth_form_tests,
    growth_form_ratio,
    GROWTH_FORM_PRESETS,
)

__version__ = "1.2.0"
__all__ = [
    "preprocess",
    "build_latlon_template",
    "load_coralnet_export",
    "compute_community_metrics",
    "run_temporal_tests",
    "run_genera_analysis",
    "prepare_growth_form_data",
    "growth_form_stats",
    "run_growth_form_tests",
    "growth_form_ratio",
    "GROWTH_FORM_PRESETS",
]
