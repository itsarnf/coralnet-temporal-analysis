"""
visualize.py
------------
Publication-ready plots for CoralNet temporal analysis.

All functions return a ``matplotlib.figure.Figure`` so they can be saved,
embedded in notebooks, or passed to further customisation.

Quick start
-----------
>>> from coralnet_temporal.visualize import (
...     plot_community_temporal,
...     plot_genera_heatmap,
...     plot_composition_change,
...     plot_transect_temporal,
... )
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
PALETTE = sns.color_palette("colorblind")


# ── community ─────────────────────────────────────────────────────────────────

def plot_community_temporal(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    month_order: list[str] | None = None,
    figsize: tuple = (16, 5),
) -> plt.Figure:
    """Boxplot + mean line for one or more community metrics over time.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``month_name`` and the metric columns.
    metrics : list[str]
        Defaults to ``['Total_Coral', 'Richness']``.
    month_order : list[str]
        Order of months along the x-axis.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if metrics is None:
        metrics = ["Total_Coral", "Richness"]

    df_plot = df.copy()
    if month_order:
        df_plot["month_name"] = pd.Categorical(
            df_plot["month_name"], categories=month_order, ordered=True
        )
    else:
        month_order = sorted(df_plot["month_name"].dropna().unique())

    colors = ["lightcoral", "lightblue", "lightgreen", "lightyellow"]
    ylabels = {
        "Total_Coral": "Hard Coral Cover (%)",
        "Richness": "Number of Genera",
        "Shannon_H": "Shannon H′",
        "Evenness": "Pielou's J",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric, color in zip(axes, metrics, colors * 10):
        sns.boxplot(
            data=df_plot, x="month_name", y=metric, ax=ax,
            color=color, width=0.6, flierprops={"alpha": 0.4},
        )
        means = df_plot.groupby("month_name")[metric].mean().reindex(month_order)
        ax.plot(range(len(month_order)), means.values, "ko-",
                linewidth=2.5, markersize=8, label="Mean", zorder=10)
        ax.set_xlabel("Month", fontsize=11)
        ax.set_ylabel(ylabels.get(metric, metric), fontsize=11)
        ax.set_title(metric.replace("_", " "), fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


# ── genera ────────────────────────────────────────────────────────────────────

def plot_genera_heatmap(
    df: pd.DataFrame,
    genera: list[str],
    month_order: list[str] | None = None,
    figsize: tuple = (14, 7),
) -> plt.Figure:
    """Heatmap of mean percent cover per genus × month.

    Parameters
    ----------
    df : pd.DataFrame
    genera : list[str]
    month_order : list[str]
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if month_order:
        months = [m for m in month_order if m in df["month_name"].unique()]
    else:
        months = sorted(df["month_name"].dropna().unique())

    matrix = pd.DataFrame(index=genera, columns=months, dtype=float)
    for genus in genera:
        for month in months:
            matrix.loc[genus, month] = (
                df[df["month_name"] == month][genus].mean()
            )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix.astype(float),
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Mean Cover (%)"},
    )
    ax.set_title("Genus Cover by Month (mean %)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Genus", fontsize=11)
    fig.tight_layout()
    return fig


def plot_top_genera_temporal(
    df: pd.DataFrame,
    genera: list[str],
    month_order: list[str] | None = None,
    figsize: tuple = (18, 10),
) -> plt.Figure:
    """Small-multiple boxplots for each genus in *genera*.

    Parameters
    ----------
    df : pd.DataFrame
    genera : list[str]
        Up to 5 genera (a 2×3 panel is created; the last cell shows total).
    month_order : list[str]
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    df_plot = df.copy()
    if month_order:
        df_plot["month_name"] = pd.Categorical(
            df_plot["month_name"], categories=month_order, ordered=True
        )
    else:
        month_order = sorted(df_plot["month_name"].dropna().unique())

    n = min(len(genera), 5)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for idx, genus in enumerate(genera[:n]):
        ax = axes[idx]
        sns.boxplot(data=df_plot, x="month_name", y=genus, ax=ax,
                    color="skyblue", width=0.6)
        means = df_plot.groupby("month_name")[genus].mean().reindex(month_order)
        ax.plot(range(len(month_order)), means.values, "ro-",
                linewidth=2, markersize=7, label="Mean", zorder=10)
        ax.set_title(genus, fontsize=12, fontweight="bold")
        ax.set_xlabel("Month", fontsize=10)
        ax.set_ylabel("% Cover", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # final panel: total coral
    ax = axes[5]
    sns.boxplot(data=df_plot, x="month_name", y="Total_Coral", ax=ax,
                color="coral", width=0.6)
    means = df_plot.groupby("month_name")["Total_Coral"].mean().reindex(month_order)
    ax.plot(range(len(month_order)), means.values, "ko-",
            linewidth=2, markersize=7, label="Mean", zorder=10)
    ax.set_title("Total Hard Coral", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel("% Cover", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # hide any unused panels
    for idx in range(n, 5):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Temporal Patterns: Top Genera + Total Coral",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return fig


# ── composition ───────────────────────────────────────────────────────────────

def plot_composition_change(
    df: pd.DataFrame,
    coral_cols: list[str],
    top_genera: list[str],
    month_a: str = "Jan",
    month_b: str = "Nov",
    figsize: tuple = (16, 5),
) -> plt.Figure:
    """Side-by-side bar comparison (top contributors + Jan vs Nov).

    Parameters
    ----------
    df : pd.DataFrame
    coral_cols : list[str]
    top_genera : list[str]
        Genera shown in the right panel.
    month_a, month_b : str
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    comp_a = df[df["month_name"] == month_a][coral_cols].mean()
    comp_b = df[df["month_name"] == month_b][coral_cols].mean()
    numerator = (comp_a - comp_b).abs().sum()
    contrib = (comp_a - comp_b).abs() / numerator * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    contrib.nlargest(8).sort_values().plot(kind="barh", ax=ax1, color="indianred")
    ax1.set_xlabel("Contribution to Dissimilarity (%)", fontsize=11)
    ax1.set_title("Top Contributors to Compositional Change", fontsize=12,
                  fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    comparison = pd.DataFrame({month_a: comp_a[top_genera],
                                month_b: comp_b[top_genera]})
    comparison.plot(kind="bar", ax=ax2, color=["steelblue", "coral"], width=0.7)
    ax2.set_ylabel("% Cover", fontsize=11)
    ax2.set_title(f"Top Genera: {month_a} vs {month_b}", fontsize=12,
                  fontweight="bold")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    ax2.legend(title="Month")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


# ── transect ──────────────────────────────────────────────────────────────────

def plot_transect_temporal(
    df: pd.DataFrame,
    metric: str = "Hard coral",
    month_order: list[str] | None = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Line plot of a metric over time, one line per transect.

    Parameters
    ----------
    df : pd.DataFrame
    metric : str
    month_order : list[str]
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if month_order is None:
        month_order = sorted(df["month_name"].dropna().unique())

    fig, ax = plt.subplots(figsize=figsize)
    for transect in sorted(df["transect"].unique()):
        sub = df[df["transect"] == transect]
        means = sub.groupby("month_name")[metric].mean().reindex(month_order)
        ax.plot(month_order, means.values, marker="o", linewidth=2,
                markersize=8, label=f"Transect {transect}")

    ax.set_xlabel("Month", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{metric} Cover (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Temporal Patterns by Transect — {metric}",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── growth forms ──────────────────────────────────────────────────────────────

def plot_growth_form_overview(
    df: "pd.DataFrame",
    form_cols: list[str],
    genus_name: str = "Genus",
    figsize: tuple = (14, 5),
) -> "plt.Figure":
    """Pie chart of overall form composition + bar chart of absolute cover.

    Mirrors the first figure in the Porites notebook.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`~coralnet_temporal.growth_forms.prepare_growth_form_data`.
    form_cols : list[str]
        Growth form column names.
    genus_name : str
        Used in titles.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    totals = df[form_cols].sum()
    props  = (totals / totals.sum() * 100).round(1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.pie(props.values, labels=form_cols, autopct="%1.1f%%",
            startangle=90, colors=PALETTE)
    ax1.set_title(f"{genus_name} Growth Form Composition", fontweight="bold")

    bars = ax2.bar(form_cols, totals.values, color=PALETTE[:len(form_cols)])
    ax2.set_ylabel("Total % Cover")
    ax2.set_title(f"{genus_name} Growth Form — Absolute Cover", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, totals.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, val,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticklabels(form_cols, rotation=20, ha="right")

    fig.tight_layout()
    return fig


def plot_growth_form_by_transect(
    df: "pd.DataFrame",
    form_cols: list[str],
    genus_name: str = "Genus",
    ratio_pair: "tuple[str, str] | None" = None,
    figsize: tuple = (16, 10),
) -> "plt.Figure":
    """Four-panel transect breakdown: stacked absolute, proportional,
    optional ratio bar, and boxplot of the dominant form.

    Parameters
    ----------
    df : pd.DataFrame
    form_cols : list[str]
    genus_name : str
    ratio_pair : (str, str), optional
        If given, the third panel shows numerator:denominator ratio bars.
        E.g. ``("Branching", "Massive")``.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    by_transect = df.groupby("transect")[form_cols].mean()
    totals      = df.groupby("transect")[form_cols].sum()
    props       = totals.div(totals.sum(axis=1), axis=0) * 100
    transects   = sorted(df["transect"].unique())
    x_labels    = [str(t) for t in transects]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # panel 1 — stacked absolute
    by_transect.plot(kind="bar", stacked=True, ax=axes[0, 0],
                     color=PALETTE[:len(form_cols)], width=0.8)
    axes[0, 0].set_title(f"Mean {genus_name} Cover by Growth Form", fontweight="bold")
    axes[0, 0].set_xlabel("Transect")
    axes[0, 0].set_ylabel("Mean % Cover")
    axes[0, 0].set_xticklabels(x_labels, rotation=0)
    axes[0, 0].legend(form_cols, loc="upper right", fontsize=8)
    axes[0, 0].grid(axis="y", alpha=0.3)

    # panel 2 — proportional
    props.plot(kind="bar", stacked=True, ax=axes[0, 1],
               color=PALETTE[:len(form_cols)], width=0.8)
    axes[0, 1].set_title("Relative Growth Form Composition", fontweight="bold")
    axes[0, 1].set_xlabel("Transect")
    axes[0, 1].set_ylabel("Proportion (%)")
    axes[0, 1].set_xticklabels(x_labels, rotation=0)
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].legend(form_cols, loc="upper right", fontsize=8)
    axes[0, 1].grid(axis="y", alpha=0.3)

    # panel 3 — ratio or fallback dominant form bar
    ax3 = axes[1, 0]
    if ratio_pair and ratio_pair[0] in form_cols and ratio_pair[1] in form_cols:
        num, den = ratio_pair
        ratios = (by_transect[num] / by_transect[den].replace(0, np.nan)).round(2)
        bars = ax3.bar(transects, ratios.values, color=PALETTE[0])
        ax3.axhline(y=1, color="red", linestyle="--", linewidth=2, label="Equal (1:1)")
        ax3.set_title(f"{num} : {den} Ratio by Transect", fontweight="bold")
        ax3.set_xlabel("Transect")
        ax3.set_ylabel(f"{num} : {den} Ratio")
        ax3.set_xticks(transects)
        ax3.set_xticklabels(x_labels)
        ax3.legend(fontsize=8)
        ax3.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, ratios.values):
            if not np.isnan(val):
                ax3.text(bar.get_x() + bar.get_width() / 2, val,
                         f"{val:.2f}:1", ha="center", va="bottom", fontsize=9)
    else:
        dominant = form_cols[0]
        ax3.bar(transects, by_transect[dominant].values, color=PALETTE[0])
        ax3.set_title(f"{dominant} Cover by Transect", fontweight="bold")
        ax3.set_xlabel("Transect")
        ax3.set_ylabel("Mean % Cover")
        ax3.set_xticks(transects)
        ax3.set_xticklabels(x_labels)
        ax3.grid(axis="y", alpha=0.3)

    # panel 4 — boxplot of dominant form
    ax4 = axes[1, 1]
    dominant = form_cols[0]
    data_by_transect = [
        df[df["transect"] == t][dominant].values for t in transects
    ]
    ax4.boxplot(data_by_transect, tick_labels=x_labels, patch_artist=True,
                showmeans=True, boxprops=dict(facecolor=PALETTE[0]))
    ax4.set_title(f"{dominant} {genus_name} Distribution by Transect",
                  fontweight="bold")
    ax4.set_xlabel("Transect")
    ax4.set_ylabel(f"{dominant} % Cover")
    ax4.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{genus_name} Growth Forms by Transect",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_growth_form_temporal(
    df: "pd.DataFrame",
    form_cols: list[str],
    genus_name: str = "Genus",
    month_order: "list[str] | None" = None,
    ratio_pair: "tuple[str, str] | None" = None,
    figsize: tuple = (16, 10),
) -> "plt.Figure":
    """Four-panel temporal plot: absolute lines, stacked area,
    proportional bars, and optional ratio line.

    Parameters
    ----------
    df : pd.DataFrame
    form_cols : list[str]
    genus_name : str
    month_order : list[str], optional
    ratio_pair : (str, str), optional
        If given, panel 4 shows the ratio over time with a 1:1 reference line.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    mn = df.groupby("month_name")[form_cols].mean()
    if month_order:
        mn = mn.reindex([m for m in month_order if m in mn.index])
    months_present = list(mn.index)

    totals = df.groupby("month_name")[form_cols].sum()
    if month_order:
        totals = totals.reindex([m for m in month_order if m in totals.index])
    props = totals.div(totals.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # panel 1 — absolute lines
    ax1 = axes[0, 0]
    for col, color in zip(form_cols, PALETTE):
        ax1.plot(months_present, mn[col].values, "o-",
                 linewidth=2.5, markersize=9, label=col, color=color)
    ax1.set_title("Temporal Patterns — Absolute Cover", fontweight="bold")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Mean % Cover")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # panel 2 — stacked area
    ax2 = axes[0, 1]
    ax2.stackplot(range(len(months_present)),
                  *[mn[col].values for col in form_cols],
                  labels=form_cols, alpha=0.7,
                  colors=PALETTE[:len(form_cols)])
    ax2.set_xticks(range(len(months_present)))
    ax2.set_xticklabels(months_present)
    ax2.set_title("Temporal Patterns — Stacked", fontweight="bold")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Mean % Cover")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # panel 3 — proportional bars
    ax3 = axes[1, 0]
    bottom = np.zeros(len(months_present))
    for col, color in zip(form_cols, PALETTE):
        vals = props[col].values
        ax3.bar(months_present, vals, bottom=bottom, label=col, color=color)
        bottom += vals
    ax3.set_title("Relative Composition Over Time", fontweight="bold")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Proportion (%)")
    ax3.set_ylim(0, 100)
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    # panel 4 — ratio over time OR dominant form line
    ax4 = axes[1, 1]
    if ratio_pair and ratio_pair[0] in form_cols and ratio_pair[1] in form_cols:
        num, den = ratio_pair
        ratios = (mn[num] / mn[den].replace(0, np.nan)).round(2)
        ax4.plot(months_present, ratios.values, "o-",
                 linewidth=2.5, markersize=9, label=f"{num}:{den} ratio",
                 color=PALETTE[0])
        ax4.axhline(y=1, color="red", linestyle="--", linewidth=2, label="Equal (1:1)")
        for i, (m, v) in enumerate(zip(months_present, ratios.values)):
            if not np.isnan(v):
                ax4.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax4.set_title(f"{num} : {den} Ratio Over Time", fontweight="bold")
        ax4.set_ylabel("Ratio")
        ax4.legend(fontsize=8)
    else:
        dominant = form_cols[0]
        ax4.plot(months_present, mn[dominant].values, "o-",
                 linewidth=2.5, markersize=9, color=PALETTE[0])
        ax4.set_title(f"{dominant} Cover Over Time", fontweight="bold")
        ax4.set_ylabel("Mean % Cover")
    ax4.set_xlabel("Month")
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f"{genus_name} Growth Forms — Temporal Patterns",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_growth_form_heatmap(
    df: "pd.DataFrame",
    form_cols: list[str],
    genus_name: str = "Genus",
    month_order: "list[str] | None" = None,
    figsize: "tuple | None" = None,
) -> "plt.Figure":
    """Transect × month heatmap for each growth form (one panel per form).

    Parameters
    ----------
    df : pd.DataFrame
    form_cols : list[str]
    genus_name : str
    month_order : list[str], optional
    figsize : tuple, optional
        Defaults to (6 × n_forms, 5).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if figsize is None:
        figsize = (6 * len(form_cols), 5)

    fig, axes = plt.subplots(1, len(form_cols), figsize=figsize)
    if len(form_cols) == 1:
        axes = [axes]

    for ax, form in zip(axes, form_cols):
        pivot = df.pivot_table(
            values=form, index="transect", columns="month_name", aggfunc="mean"
        )
        if month_order:
            cols_present = [m for m in month_order if m in pivot.columns]
            pivot = pivot[cols_present]
        sns.heatmap(
            pivot, annot=True, fmt=".2f", ax=ax,
            cbar_kws={"label": "% Cover"},
            linewidths=1, linecolor="white",
        )
        ax.set_title(f"{form} {genus_name}", fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Transect")
        n_transects = pivot.shape[0]
        ax.set_yticklabels([str(i + 1) for i in range(n_transects)], rotation=0)

    fig.tight_layout()
    return fig
