"""
Plotting Module
---------------
Generates dashboard-ready figures from saved pipeline output. Reads ONLY
from `analysis_output/<dataset_name>/` (see utils/results_exporter.py) —
never re-runs Layer 0-3 — so figures can be regenerated or restyled at
any time without recomputing any analysis.

Figure groups:
    Variable overlays & distributions (raw data, any layer's baseline):
        - Per-variable time series: individual runs faint, mean/median
          prominent, std band shaded.
        - Per-variable pooled histogram + KDE.

    Correlation & clustering (Layer 2):
        - Correlation heatmap (top pairs only, for a quick glance).
        - Clustermap: dendrogram + reordered heatmap, built from Layer 2's
          own cluster_columns() output (same clustering the rest of the
          pipeline reports — not recomputed here).

    Causal structure (Layer 2, per-run PCMCI aggregated):
        - Link frequency bar chart (robustness across runs).
        - Native Tigramite causal graph / time-series graph (consensus
          links only).
        - Lag profiles (frequency + strength per lag, per top pair).

    Drift & anomaly detection (Layer 3):
        - Anomaly score ranking across runs.
        - Per-run drift breakdown by perspective (mean/variance/
          correlation/onset).
        - CUSUM onset/propagation timeline — since-when and in-what-order
          a run's variables started drifting.
        - Statistical (z-score) drift bar for a run's top drifted variables.
        - Per-variable raw trace with the detected onset point marked, for
          the small set of variables that actually drifted in a run.

Color: status colors (good/warning/serious/critical) are reserved for
severity encoding and never reused for series identity; categorical
colors are assigned in a fixed order and never cycled past what a chart
can hold. Values are computed, not eyeballed — see the reference default
palette this was built against. All figures render at a light "card"
surface (white) sized for embedding in a dashboard.

Usage (in-pipeline, right after Layer 0-3 + ResultsExporter):
    plotter = PipelinePlotter(output_dir="analysis_output/<name>/figures")
    plotter.generate_all(runs=runs, columns=columns, ...)

Standalone (reload from a saved dataset folder — no re-analysis needed):
    python -m tpm.utils.plotting analysis_output/<dataset_name>
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Patch

from scipy.cluster.hierarchy import dendrogram
from scipy.stats import gaussian_kde

try:
    import tigramite.plotting as tp
except ImportError:
    tp = None


# ═══════════════════════════════════════════════════════════════════════
# Palette — status colors reserved for severity; categorical for identity.
# Values match the project's validated default (see dataviz skill /
# references/palette.md); swap here to re-theme everything at once.
# ═══════════════════════════════════════════════════════════════════════

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}
CATEGORICAL = [
    "#2a78d6",  # 1 blue    — primary series / emphasis
    "#eb6834",  # 2 orange  — secondary series
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
BASELINE_COLOR = "#c3c2b7"

_SEQ_BLUE_STEPS = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
]
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("seq_blue", _SEQ_BLUE_STEPS)


def _status_for_score(score: float, bands: Tuple[float, float, float] = (0.15, 0.35, 0.6)) -> str:
    """
    Map a 0-1 composite score to a status color via fixed visualization
    bands. These bands are a chart-legend convention (for "how alarming
    should this look"), not a statistical threshold — the underlying
    Layer 3 detection math has its own, separately-documented, formula-
    derived thresholds (see layer3_drift.py).
    """
    lo, mid, hi = bands
    if score < lo:
        return STATUS["good"]
    if score < mid:
        return STATUS["warning"]
    if score < hi:
        return STATUS["serious"]
    return STATUS["critical"]


class PipelinePlotter:
    """
    Generates all pipeline figures from saved analysis output.

    Figures are saved as PNG files under output_dir. A dashboard reads
    and embeds them (e.g. as base64).
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _path(self, *parts: str) -> str:
        """Build a path under output_dir, creating parent dirs as needed."""
        filepath = os.path.join(self.output_dir, *parts)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        return filepath

    def _save_fig(self, fig, *path_parts: str, dpi: int = 120) -> str:
        filepath = self._path(*path_parts)
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return filepath

    @staticmethod
    def _safe_name(name: str) -> str:
        return str(name).replace("/", "_").replace(" ", "_").replace(".", "_")

    @staticmethod
    def _style_axes(ax, x_grid: bool = False, y_grid: bool = True):
        """Shared hairline/recessive chrome: thin spines, light grid, muted ticks."""
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.6)
            ax.spines[side].set_color(BASELINE_COLOR)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        if y_grid:
            ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
        if x_grid:
            ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

    @staticmethod
    def _lookup_run(mapping: Dict[Any, Any], run_id: Any) -> Any:
        """
        Look up run_id in a dict that may be keyed by int or by str.

        run_summary is always str-keyed (Layer 3 / JSON round-trip
        always stringifies), while `runs` (raw per-run DataFrames) is
        typically int-keyed. Plotting functions pass run_ids around
        between both, so every cross-dict lookup goes through here
        instead of ad hoc .get(str(x), .get(x)) pairs that only cover
        one direction.
        """
        if run_id in mapping:
            return mapping[run_id]
        if str(run_id) in mapping:
            return mapping[str(run_id)]
        try:
            as_int = int(run_id)
        except (TypeError, ValueError):
            return None
        return mapping.get(as_int)

    # ═══════════════════════════════════════════════════════════════════
    # Variable overlay & distribution plots
    # ═══════════════════════════════════════════════════════════════════

    def plot_per_variable_overlays(
        self,
        runs: Dict[Any, pd.DataFrame],
        columns: List[str],
    ) -> Dict[str, str]:
        """
        Plot per-variable time series with run overlays.

        Individual runs are drawn in one neutral, low-alpha color (not
        cycled per run — with many runs, per-run hue would be both
        illegible and meaningless, since no one can read "run 237 is the
        teal one"). The mean trajectory and std band carry the real
        signal.

        Returns
        -------
        dict
            {column_name: filepath} of per-variable overlay PNGs.
        """
        run_ids = sorted(runs.keys())
        n_runs = len(run_ids)

        valid_cols = [c for c in columns if all(c in runs[rid].columns for rid in run_ids)]
        if not valid_cols:
            return {}

        result = {}
        for col in valid_cols:
            fig, ax = plt.subplots(figsize=(3.5, 1.4))

            all_series = [runs[rid][col].values for rid in run_ids]
            for series in all_series:
                alpha = 0.12 if n_runs > 1 else 0.7
                ax.plot(series, color=BASELINE_COLOR, alpha=alpha, linewidth=0.6, zorder=1)

            min_len = min(len(s) for s in all_series)
            aligned = np.array([s[:min_len] for s in all_series])
            mean_traj = np.mean(aligned, axis=0)
            std_traj = np.std(aligned, axis=0)
            median_traj = np.median(aligned, axis=0)
            x = np.arange(min_len)

            if n_runs > 1:
                ax.fill_between(
                    x, mean_traj - std_traj, mean_traj + std_traj,
                    alpha=0.18, color=CATEGORICAL[0], zorder=2, linewidth=0,
                )
            ax.plot(x, mean_traj, color=INK_PRIMARY, linewidth=1.4, zorder=5)
            ax.plot(x, median_traj, color=CATEGORICAL[1], linewidth=0.9,
                    linestyle="--", alpha=0.8, zorder=4)

            ax.tick_params(labelsize=5, colors=INK_SECONDARY)
            ax.set_xticks([])
            ax.grid(True, color=GRID_COLOR, linewidth=0.5)
            ax.margins(x=0.01, y=0.05)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color(BASELINE_COLOR)

            plt.tight_layout(pad=0.3)
            filepath = self._save_fig(
                fig, "var_overlays", f"{self._safe_name(col)}.png",
                dpi=100,
            )
            result[col] = filepath

        return result

    def plot_per_variable_histograms(
        self,
        runs: Dict[Any, pd.DataFrame],
        columns: List[str],
    ) -> Dict[str, str]:
        """
        Plot per-variable distribution histograms pooled across all runs,
        with a KDE overlay and mean/median reference lines.

        Returns
        -------
        dict
            {column_name: filepath} of per-variable histogram PNGs.
        """
        run_ids = sorted(runs.keys())
        valid_cols = [c for c in columns if all(c in runs[rid].columns for rid in run_ids)]
        if not valid_cols:
            return {}

        result = {}
        for col in valid_cols:
            pooled = pd.concat([runs[rid][col].dropna() for rid in run_ids], ignore_index=True)
            if len(pooled) == 0:
                continue

            fig, ax = plt.subplots(figsize=(3.6, 2.1))
            n_bins = min(50, max(10, len(pooled) // 50))
            ax.hist(
                pooled.values, bins=n_bins, color=CATEGORICAL[0], alpha=0.55,
                edgecolor="white", linewidth=0.3, density=True, zorder=2,
            )

            try:
                kde_x = np.linspace(pooled.min(), pooled.max(), 200)
                kde = gaussian_kde(pooled.values)
                ax.plot(kde_x, kde(kde_x), color=INK_PRIMARY, linewidth=1.3, zorder=5)
            except Exception:
                pass

            ax.axvline(pooled.mean(), color=INK_PRIMARY, linewidth=1.1, zorder=6)
            ax.axvline(pooled.median(), color=CATEGORICAL[1], linewidth=1.0,
                       linestyle="--", alpha=0.8, zorder=6)

            ax.tick_params(labelsize=8, colors=INK_SECONDARY)
            ax.set_yticks([])
            ax.grid(axis="x", color=GRID_COLOR, linewidth=0.5)
            ax.margins(x=0.01, y=0.05)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_linewidth(0.5)
            ax.spines["bottom"].set_color(BASELINE_COLOR)

            plt.tight_layout(pad=0.3)
            filepath = self._save_fig(
                fig, "var_histograms", f"{self._safe_name(col)}.png",
                dpi=130,
            )
            result[col] = filepath

        return result

    # ═══════════════════════════════════════════════════════════════════
    # Correlation & clustering (Layer 2)
    # ═══════════════════════════════════════════════════════════════════

    def _filter_top_corr(self, corr_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
        """Select top_n variables with strongest average correlations."""
        abs_corr = corr_df.abs()
        np.fill_diagonal(abs_corr.values, 0)
        mean_corr = abs_corr.mean(axis=1)
        top_cols = mean_corr.nlargest(top_n).index.tolist()
        return corr_df.loc[top_cols, top_cols]

    def plot_correlation_heatmap(
        self,
        corr_df: pd.DataFrame,
        method: str = "pearson",
        top_n: int = 30,
    ) -> Optional[str]:
        """
        Quick-glance correlation heatmap, filtered to the top_n variables
        with the strongest average |correlation| (for readability). For
        the full clustered structure, see plot_correlation_clustermap.
        """
        if corr_df is None or len(corr_df) < 2:
            return None

        if len(corr_df) > top_n:
            corr_df = self._filter_top_corr(corr_df, top_n)

        n = len(corr_df)
        figsize = max(8, n * 0.3)
        fig, ax = plt.subplots(figsize=(figsize, figsize))

        data = np.nan_to_num(corr_df.values.astype(float), nan=0.0)
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        im = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="equal")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr_df.columns, rotation=90, fontsize=7, color=INK_SECONDARY)
        ax.set_yticklabels(corr_df.columns, fontsize=7, color=INK_SECONDARY)

        plt.colorbar(im, ax=ax, shrink=0.8, label=f"{method.title()} correlation")
        ax.set_title(f"{method.title()} Correlation — Top {n} Variables", fontsize=12, pad=10, color=INK_PRIMARY)

        plt.tight_layout()
        return self._save_fig(fig, f"corr_heatmap_{method}.png")

    def plot_correlation_clustermap(
        self,
        corr_df: pd.DataFrame,
        clusters: Optional[Dict[str, Any]],
        method: str = "pearson",
    ) -> Optional[str]:
        """
        Hierarchical clustering dendrogram, leaves colored by cluster
        assignment — built from Layer 2's own cluster_columns() output
        (the same clustering reported elsewhere in the pipeline, not a
        separately recomputed one).

        Deliberately NOT another heatmap: plot_correlation_heatmap
        already shows the raw correlation values, so pairing it with a
        second heatmap here (as an earlier version of this plot did, via
        seaborn's clustermap) just repeats the same numbers a second
        time at a smaller, harder-to-read scale. What a plain heatmap
        *can't* show is the merge hierarchy and which cluster each
        variable landed in — that's what this plot is for.

        Parameters
        ----------
        corr_df : pd.DataFrame
            Pooled correlation matrix — only used to confirm which
            clustered columns are actually present in it.
        clusters : dict
            Output of CausalAnalyzer.cluster_columns() — needs
            "linkage_matrix", "columns", "column_cluster".

        Returns
        -------
        str or None
        """
        if not clusters or clusters.get("status") != "ok" or corr_df is None:
            return None

        cols = clusters["columns"]
        cols = [c for c in cols if c in corr_df.columns]
        if len(cols) < 3:
            return None

        linkage_matrix = np.asarray(clusters["linkage_matrix"])
        column_cluster = clusters.get("column_cluster", {})

        # Fixed-order categorical colors per cluster id, the same order
        # used for cluster tags elsewhere in the dashboard, so a cluster
        # reads as the same color everywhere; beyond 8 distinct clusters,
        # fold the rest into one shared muted "other" bucket rather than
        # cycling hues past what a legend can hold (they'd otherwise all
        # render as the same gray while the legend claimed they were
        # distinct clusters).
        cluster_ids = sorted(set(column_cluster.get(c) for c in cols if c in column_cluster))
        primary_ids = cluster_ids[:len(CATEGORICAL)]
        overflow_ids = cluster_ids[len(CATEGORICAL):]
        color_for_cluster = {cid: CATEGORICAL[i] for i, cid in enumerate(primary_ids)}
        color_for_cluster.update({cid: INK_MUTED for cid in overflow_ids})

        n = len(cols)
        fig, ax = plt.subplots(figsize=(9, max(6, n * 0.28)))
        dendrogram(
            linkage_matrix, labels=cols, orientation="left", ax=ax,
            color_threshold=0, above_threshold_color=BASELINE_COLOR,
        )
        ax.set_xlabel(f"Cluster distance (from {method} correlation)", fontsize=10, color=INK_SECONDARY)
        ax.set_xticks([])
        ax.grid(False)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)

        # Color each leaf's label by its cluster assignment — the one
        # thing a plain dendrogram doesn't already convey on its own.
        for label in ax.get_ymajorticklabels():
            cid = column_cluster.get(label.get_text())
            label.set_color(color_for_cluster.get(cid, INK_MUTED))
            label.set_fontsize(8)

        if primary_ids:
            handles = [
                Patch(facecolor=color_for_cluster[cid], label=f"Cluster {cid}")
                for cid in primary_ids
            ]
            if overflow_ids:
                handles.append(Patch(facecolor=INK_MUTED, label="Other clusters"))
            ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False, ncol=min(4, len(handles)))

        ax.set_title(
            f"Variable Clusters (k={clusters.get('best_k')}, "
            f"silhouette={clusters.get('best_silhouette')}) — hierarchical structure",
            fontsize=12, color=INK_PRIMARY, pad=10,
        )

        plt.tight_layout()
        return self._save_fig(fig, f"clustermap_{method}.png")

    # ═══════════════════════════════════════════════════════════════════
    # Causal structure (Layer 2, per-run PCMCI aggregated)
    # ═══════════════════════════════════════════════════════════════════

    def plot_causal_frequency(
        self,
        aggregated: Dict[str, Any],
        top_n: int = 20,
    ) -> Optional[str]:
        """
        Bar chart of causal link frequency across runs — how many runs
        each directed link X(t-tau) -> Y was detected as significant.

        Frequency is a continuous magnitude (0..n_runs), so bars are
        colored via a single-hue sequential ramp rather than discrete
        arbitrary tiers.
        """
        link_freq = aggregated.get("link_frequency", {})
        n_runs = aggregated.get("n_runs", 1)
        if not link_freq:
            return None

        sorted_links = sorted(link_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        if not sorted_links:
            return None

        labels, counts = [], []
        for key, count in sorted_links:
            src, tgt, tau = key.split("|")
            labels.append(f"{src}(t-{tau}) → {tgt}")
            counts.append(count)

        fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.35)))
        y_pos = range(len(labels))
        norm = plt.Normalize(0, n_runs)
        colors = [SEQUENTIAL_BLUE(norm(c)) for c in counts]
        bars = ax.barh(y_pos, counts, color=colors, edgecolor="white", linewidth=0.5, zorder=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8, fontfamily="monospace", color=INK_SECONDARY)
        ax.set_xlabel(f"Runs detected (out of {n_runs})", fontsize=10, color=INK_SECONDARY)
        ax.set_title(f"Top {len(labels)} Causal Links by Robustness Across Runs", fontsize=12, pad=10, color=INK_PRIMARY)
        ax.invert_yaxis()
        ax.set_xlim(0, n_runs + 0.5)
        self._style_axes(ax, x_grid=True, y_grid=False)

        for bar, count in zip(bars, counts):
            pct = count / n_runs * 100
            ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                    f"{count}/{n_runs} ({pct:.0f}%)", va="center", fontsize=7, color=INK_SECONDARY)

        sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL_BLUE, norm=norm)
        plt.colorbar(sm, ax=ax, shrink=0.6, label="Runs detected", pad=0.02)

        plt.tight_layout()
        return self._save_fig(fig, "causal_frequency.png")

    def _build_consensus_arrays(
        self,
        aggregated: Dict[str, Any],
        min_abs_val: float = 0.0,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, List[str], List[int]]]:
        """
        Build Tigramite-compatible graph/val_matrix arrays from
        aggregated per-run results, keeping only consensus links
        (detected in ALL runs) and only variables that participate in
        at least one surviving link.
        """
        link_freq = aggregated.get("link_frequency", {})
        var_names = aggregated.get("var_names", [])
        n_runs = aggregated.get("n_runs", 1)
        mean_vals = aggregated.get("mean_val", {})
        tau_max = aggregated.get("metadata", {}).get("parameters", {}).get("tau_max", 10)

        if not link_freq or not var_names or n_runs < 1:
            return None

        name_to_idx = {v: i for i, v in enumerate(var_names)}
        consensus = {
            key: count for key, count in link_freq.items()
            if count >= n_runs and abs(mean_vals.get(key, 0)) >= min_abs_val
        }
        if not consensus:
            return None

        involved = set()
        for key in consensus:
            src, tgt, _tau = key.split("|")
            involved.update((src, tgt))
        if not involved:
            return None

        filtered_names = [v for v in var_names if v in involved]
        filtered_indices = [name_to_idx[v] for v in filtered_names]
        n = len(filtered_names)
        new_idx = {v: i for i, v in enumerate(filtered_names)}

        graph = np.zeros((n, n, tau_max + 1), dtype='<U3')
        val_matrix = np.zeros((n, n, tau_max + 1))

        lag0_links: Dict[Tuple[int, int], list] = {}
        for key in consensus:
            src, tgt, tau_str = key.split("|")
            tau = int(tau_str)
            if src not in new_idx or tgt not in new_idx or tau > tau_max:
                continue
            i, j = new_idx[src], new_idx[tgt]
            if tau == 0:
                pair = (min(i, j), max(i, j))
                lag0_links.setdefault(pair, []).append((i, j, key))
            else:
                graph[i, j, tau] = "-->"
                val_matrix[i, j, tau] = mean_vals.get(key, 0)

        for _pair, entries in lag0_links.items():
            if len(entries) >= 2:
                for i, j, key in entries:
                    graph[i, j, 0] = "o-o"
                    val_matrix[i, j, 0] = mean_vals.get(key, 0)
            else:
                i, j, key = entries[0]
                graph[i, j, 0] = "-->"
                val_matrix[i, j, 0] = mean_vals.get(key, 0)

        for v in filtered_names:
            i = new_idx[v]
            best_auto = 0.0
            for tau in range(1, tau_max + 1):
                auto_val = mean_vals.get(f"{v}|{v}|{tau}", 0)
                if abs(auto_val) > abs(best_auto):
                    best_auto = auto_val
            val_matrix[i, i, 0] = best_auto

        return graph, val_matrix, filtered_names, filtered_indices

    def plot_causal_graph(self, aggregated: Dict[str, Any], min_abs_val: float = 0.5) -> Optional[str]:
        """Causal network via Tigramite's native tp.plot_graph() — consensus links only."""
        if tp is None:
            return None
        result = self._build_consensus_arrays(aggregated, min_abs_val=min_abs_val)
        if result is None:
            return None

        graph, val_matrix, filtered_names, _ = result
        n = len(filtered_names)
        n_runs = aggregated.get("n_runs", 1)
        n_links = int(np.sum(graph != ""))

        fig, ax = plt.subplots(figsize=(max(10, n * 0.5), max(8, n * 0.4)))
        tp.plot_graph(
            graph=graph, val_matrix=val_matrix, var_names=filtered_names, fig_ax=(fig, ax),
            link_colorbar_label="Mean Partial Corr", node_colorbar_label="Max Auto-dependency",
            arrow_linewidth=6.0, node_size=0.4, arrowhead_size=16, curved_radius=0.25,
            label_fontsize=10, node_label_size=9, link_label_fontsize=8,
            vmin_edges=-0.5, vmax_edges=0.5, edge_ticks=0.25, show_autodependency_lags=True,
        )
        ax.set_title(
            f"Causal Graph — Consensus Links (in all {n_runs} runs, |parcorr| ≥ {min_abs_val}, "
            f"{n_links} links, {n} variables)", fontsize=12, pad=15, color=INK_PRIMARY,
        )
        return self._save_fig(fig, "causal_graph.png")

    def plot_causal_time_series_graph(self, aggregated: Dict[str, Any], min_abs_val: float = 0.5) -> Optional[str]:
        """Time-series causal graph via Tigramite's native tp.plot_time_series_graph()."""
        if tp is None:
            return None
        result = self._build_consensus_arrays(aggregated, min_abs_val=min_abs_val)
        if result is None:
            return None

        graph, val_matrix, filtered_names, _ = result
        n = len(filtered_names)
        n_runs = aggregated.get("n_runs", 1)
        n_links = int(np.sum(graph != ""))

        fig, ax = plt.subplots(figsize=(max(12, n * 0.6), max(6, n * 0.5)))
        tp.plot_time_series_graph(
            graph=graph, val_matrix=val_matrix, var_names=filtered_names, fig_ax=(fig, ax),
            link_colorbar_label="Mean Partial Corr", arrow_linewidth=4.0, node_size=0.08,
            arrowhead_size=14, curved_radius=0.2, label_fontsize=9,
            vmin_edges=-0.5, vmax_edges=0.5, edge_ticks=0.25,
        )
        ax.set_title(
            f"Time-Series Causal Graph — Consensus Links (in all {n_runs} runs, "
            f"|parcorr| ≥ {min_abs_val}, {n_links} links, {n} variables)",
            fontsize=12, pad=15, color=INK_PRIMARY,
        )
        return self._save_fig(fig, "causal_time_series.png")

    def plot_causal_lag_profile(self, aggregated: Dict[str, Any], top_n: int = 10) -> Optional[str]:
        """
        For each of the top_n most frequent links, show frequency and
        mean strength per lag — as two stacked single-axis panels per
        pair (not a dual-axis overlay: two different scales sharing one
        axis invents a visual correlation that isn't in the data).
        """
        lag_profiles = aggregated.get("lag_profiles", {})
        n_runs = aggregated.get("n_runs", 1)
        if not lag_profiles:
            return None

        pair_scores = {key: max(p.get("frequencies", [0])) for key, p in lag_profiles.items()}
        top_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        if not top_pairs:
            return None

        n_pairs = len(top_pairs)
        n_cols_grid = min(3, n_pairs)
        n_rows_grid = int(np.ceil(n_pairs / n_cols_grid))

        fig, axes = plt.subplots(
            n_rows_grid * 2, n_cols_grid,
            figsize=(4.5 * n_cols_grid, 3.6 * n_rows_grid),
            squeeze=False, gridspec_kw={"height_ratios": [1.3, 1] * n_rows_grid, "hspace": 0.55},
        )

        norm = plt.Normalize(0, n_runs)
        for idx, (pair_key, _) in enumerate(top_pairs):
            grid_row, grid_col = idx // n_cols_grid, idx % n_cols_grid
            ax_freq = axes[grid_row * 2][grid_col]
            ax_val = axes[grid_row * 2 + 1][grid_col]

            profile = lag_profiles[pair_key]
            taus = profile.get("taus", [])
            mean_vals = profile.get("mean_vals", [])
            std_vals = profile.get("std_vals", [])
            frequencies = profile.get("frequencies", [])
            src, tgt = pair_key.split("|")

            colors = [SEQUENTIAL_BLUE(norm(f)) for f in frequencies]
            ax_freq.bar(taus, frequencies, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
            ax_freq.set_title(f"{src} → {tgt}", fontsize=9, fontweight="bold", color=INK_PRIMARY)
            ax_freq.set_ylabel("# Runs", fontsize=7, color=INK_SECONDARY)
            ax_freq.set_ylim(0, n_runs + 0.5)
            ax_freq.set_xticks(taus)
            ax_freq.tick_params(labelsize=6, colors=INK_SECONDARY, labelbottom=False)
            self._style_axes(ax_freq)

            if mean_vals:
                ax_val.plot(taus, mean_vals, color=CATEGORICAL[1], marker="o", markersize=3, linewidth=1.2, zorder=3)
                if std_vals:
                    ax_val.fill_between(
                        taus, [m - s for m, s in zip(mean_vals, std_vals)],
                        [m + s for m, s in zip(mean_vals, std_vals)],
                        alpha=0.15, color=CATEGORICAL[1], linewidth=0, zorder=2,
                    )
            ax_val.set_xlabel("Lag", fontsize=7, color=INK_SECONDARY)
            ax_val.set_ylabel("Mean |parcorr|", fontsize=7, color=INK_SECONDARY)
            ax_val.set_xticks(taus)
            ax_val.tick_params(labelsize=6, colors=INK_SECONDARY)
            self._style_axes(ax_val)

        for idx in range(n_pairs, n_rows_grid * n_cols_grid):
            grid_row, grid_col = idx // n_cols_grid, idx % n_cols_grid
            axes[grid_row * 2][grid_col].set_visible(False)
            axes[grid_row * 2 + 1][grid_col].set_visible(False)

        fig.suptitle(
            f"Lag Profiles — Top {n_pairs} Causal Pairs (top: run count, bottom: mean |parcorr|)",
            fontsize=12, fontweight="bold", y=1.01, color=INK_PRIMARY,
        )
        return self._save_fig(fig, "causal_lag_profiles.png")

    # ═══════════════════════════════════════════════════════════════════
    # Drift & anomaly detection (Layer 3)
    # ═══════════════════════════════════════════════════════════════════

    def plot_anomaly_ranking(
        self,
        run_summary: Dict[str, Any],
        top_n: int = 20,
    ) -> Optional[str]:
        """
        Rank scored runs by composite anomaly score. Bar color is a
        fixed status band (good/warning/serious/critical) — a chart
        convention for "how alarming," not a statistical threshold.
        """
        if not run_summary:
            return None

        ranked = sorted(run_summary.items(), key=lambda kv: kv[1].get("anomaly_score", 0), reverse=True)[:top_n]
        if not ranked:
            return None

        labels = [str(rid) for rid, _ in ranked]
        scores = [rs.get("anomaly_score", 0) for _rid, rs in ranked]
        colors = [_status_for_score(s) for s in scores]

        fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.35)))
        y_pos = range(len(labels))
        bars = ax.barh(y_pos, scores, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Run {l}" for l in labels], fontsize=8, color=INK_SECONDARY)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Anomaly score (0–1)", fontsize=10, color=INK_SECONDARY)
        ax.set_title(f"Anomaly Score Ranking — Top {len(labels)} of {len(run_summary)} Run(s)",
                     fontsize=12, pad=10, color=INK_PRIMARY)
        self._style_axes(ax, x_grid=True, y_grid=False)

        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}", va="center", fontsize=7, color=INK_SECONDARY)

        legend_elements = [
            Patch(facecolor=STATUS["good"], label="Good (<0.15)"),
            Patch(facecolor=STATUS["warning"], label="Warning (0.15–0.35)"),
            Patch(facecolor=STATUS["serious"], label="Serious (0.35–0.6)"),
            Patch(facecolor=STATUS["critical"], label="Critical (≥0.6)"),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="lower right", frameon=False)

        plt.tight_layout()
        return self._save_fig(fig, "layer3", "anomaly_ranking.png")

    def plot_run_drift_breakdown(
        self,
        run_summary: Dict[str, Any],
        run_ids: List[Any],
    ) -> Optional[str]:
        """
        For each of the given runs, a small bar chart of the four drift
        perspectives' fractions (mean shift / variance / correlation /
        onset) — shows WHICH perspective is driving a run's anomaly
        score, not just the composite number.
        """
        run_ids = [rid for rid in run_ids if self._lookup_run(run_summary, rid) is not None]
        if not run_ids:
            return None

        n = len(run_ids)
        n_cols_grid = min(4, n)
        n_rows_grid = int(np.ceil(n / n_cols_grid))
        fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(3.2 * n_cols_grid, 3 * n_rows_grid), squeeze=False)

        categories = ["Mean\nshift", "Variance", "Correlation", "Onset\n(CUSUM)"]
        for idx, rid in enumerate(run_ids):
            rs = self._lookup_run(run_summary, rid) or {}
            row, col = idx // n_cols_grid, idx % n_cols_grid
            ax = axes[row][col]

            values = [
                rs.get("frac_mean_drifted", 0), rs.get("frac_var_drifted", 0),
                rs.get("frac_corr_drifted", 0), rs.get("frac_onset_detected", 0),
            ]
            ax.bar(categories, values, color=CATEGORICAL[0], edgecolor="white", linewidth=0.5, zorder=3)
            ax.set_ylim(0, 1.0)
            ax.set_title(f"Run {rid} (score={rs.get('anomaly_score', 0):.3f})", fontsize=9, color=INK_PRIMARY)
            ax.tick_params(labelsize=7, colors=INK_SECONDARY)
            self._style_axes(ax)
            for i, v in enumerate(values):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=INK_SECONDARY)

        for idx in range(n, n_rows_grid * n_cols_grid):
            row, col = idx // n_cols_grid, idx % n_cols_grid
            axes[row][col].set_visible(False)

        fig.suptitle("Per-Run Drift Breakdown by Perspective", fontsize=12, fontweight="bold", y=1.02, color=INK_PRIMARY)
        plt.tight_layout()
        return self._save_fig(fig, "layer3", "drift_breakdown.png")

    def plot_cusum_propagation(
        self,
        run_summary: Dict[str, Any],
        run_id: Any,
    ) -> Optional[str]:
        """
        Onset/propagation timeline for one run: which variable started
        drifting first, and in what order — the direct visualization of
        Layer 3's CUSUM perspective (layer3_drift.py's "since which
        sample" answer). One marker per variable at its onset sample,
        ordered earliest-first, connected by a thin cascade guide line.
        """
        rs = self._lookup_run(run_summary, run_id) or {}
        prop = rs.get("propagation_order", [])
        if not prop:
            return None

        variables = [p["variable"] for p in prop]
        onsets = [p["onset_sample"] for p in prop]
        y_pos = list(range(len(variables)))

        fig, ax = plt.subplots(figsize=(9, max(3, len(variables) * 0.4)))
        ax.plot(onsets, y_pos, color=BASELINE_COLOR, linewidth=1.0, zorder=2)
        ax.hlines(y_pos, xmin=0, xmax=onsets, color=GRID_COLOR, linewidth=1.0, zorder=1)
        ax.scatter(onsets, y_pos, color=CATEGORICAL[1], s=90, zorder=5,
                   edgecolors="white", linewidths=1.0)

        for i, (var, onset) in enumerate(zip(variables, onsets)):
            ax.annotate(f"  sample {onset}", (onset, i), fontsize=7, va="center", color=INK_SECONDARY)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(variables, fontsize=9, color=INK_SECONDARY)
        ax.invert_yaxis()
        ax.set_xlabel("Sample index (onset detected by CUSUM)", fontsize=10, color=INK_SECONDARY)
        ax.set_title(
            f"Run {run_id} — Onset & Propagation Order ({len(variables)} variable(s) drifted, "
            f"earliest first)", fontsize=12, pad=10, color=INK_PRIMARY,
        )
        ax.set_xlim(left=0)
        self._style_axes(ax, x_grid=True, y_grid=False)

        plt.tight_layout()
        return self._save_fig(fig, "layer3", f"run_{run_id}", "cusum_propagation.png")

    def plot_statistical_drift_bar(
        self,
        run_summary: Dict[str, Any],
        run_id: Any,
    ) -> Optional[str]:
        """
        Bar chart of the top mean-drifted variables for one run (from
        run_summary's top_mean_drifted, already sorted by |z-score|).
        """
        rs = self._lookup_run(run_summary, run_id) or {}
        top = rs.get("top_mean_drifted", [])
        if not top:
            return None

        labels = [t["variable"] for t in top]
        z_scores = [t["z_score"] for t in top]
        colors = [STATUS["critical"] if abs(z) >= 3 else STATUS["warning"] for z in z_scores]

        fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.4)))
        y_pos = range(len(labels))
        ax.barh(y_pos, z_scores, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
        ax.axvline(0, color=BASELINE_COLOR, linewidth=0.8, zorder=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9, color=INK_SECONDARY)
        ax.invert_yaxis()
        ax.set_xlabel("Mean-shift z-score vs. baseline", fontsize=10, color=INK_SECONDARY)
        ax.set_title(f"Run {run_id} — Top Mean-Drifted Variables", fontsize=12, pad=10, color=INK_PRIMARY)
        self._style_axes(ax, x_grid=True, y_grid=False)

        plt.tight_layout()
        return self._save_fig(fig, "layer3", f"run_{run_id}", "statistical_drift.png")

    def plot_variable_with_onset(
        self,
        runs: Dict[Any, pd.DataFrame],
        run_id: Any,
        baseline: Dict[str, Any],
        run_summary: Dict[str, Any],
        max_vars: int = 6,
    ) -> Optional[str]:
        """
        For the run's drifted variables (from propagation_order, capped
        at max_vars, earliest first), plot the actual raw trace against
        the baseline's normal range, with the detected onset marked —
        the "dig deeper" companion to plot_cusum_propagation.
        """
        rs = self._lookup_run(run_summary, run_id) or {}
        prop = rs.get("propagation_order", [])[:max_vars]
        run_df = self._lookup_run(runs, run_id)
        if run_df is None or not prop:
            return None

        n = len(prop)
        n_cols_grid = min(3, n)
        n_rows_grid = int(np.ceil(n / n_cols_grid))
        fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(4.5 * n_cols_grid, 2.6 * n_rows_grid), squeeze=False)

        for idx, p in enumerate(prop):
            col, onset = p["variable"], p["onset_sample"]
            row, grid_col = idx // n_cols_grid, idx % n_cols_grid
            ax = axes[row][grid_col]

            if col not in run_df.columns:
                ax.set_visible(False)
                continue

            series = run_df[col].values
            x = np.arange(len(series))
            bl = baseline.get(col, {})
            mu, sigma = bl.get("pooled_mean"), bl.get("pooled_std")

            ax.plot(x, series, color=CATEGORICAL[0], linewidth=1.0, zorder=4)
            if mu is not None and sigma is not None:
                ax.axhline(mu, color=BASELINE_COLOR, linewidth=0.8, zorder=2)
                ax.fill_between(x, mu - 2 * sigma, mu + 2 * sigma, color=BASELINE_COLOR, alpha=0.15, zorder=1, linewidth=0)

            if onset is not None:
                ax.axvline(onset, color=STATUS["critical"], linewidth=1.2, linestyle="-", zorder=5)
                ax.axvspan(onset, len(series), color=STATUS["critical"], alpha=0.06, zorder=0)

            ax.set_title(f"{col} (onset @ {onset})", fontsize=9, color=INK_PRIMARY)
            ax.tick_params(labelsize=6, colors=INK_SECONDARY)
            self._style_axes(ax)

        for idx in range(n, n_rows_grid * n_cols_grid):
            row, grid_col = idx // n_cols_grid, idx % n_cols_grid
            axes[row][grid_col].set_visible(False)

        fig.suptitle(
            f"Run {run_id} — Drifted Variables with Detected Onset "
            f"(shaded band = baseline mean ± 2σ)",
            fontsize=12, fontweight="bold", y=1.02, color=INK_PRIMARY,
        )
        plt.tight_layout()
        return self._save_fig(fig, "layer3", f"run_{run_id}", "variables_with_onset.png")

    # ═══════════════════════════════════════════════════════════════════
    # Generate all figures
    # ═══════════════════════════════════════════════════════════════════

    def generate_all(
        self,
        runs: Optional[Dict[Any, pd.DataFrame]] = None,
        columns: Optional[List[str]] = None,
        pooled_correlations: Optional[Dict[str, pd.DataFrame]] = None,
        corr_methods: Optional[List[str]] = None,
        clusters: Optional[Dict[str, Any]] = None,
        causal_aggregated: Optional[Dict[str, Any]] = None,
        run_summary: Optional[Dict[str, Any]] = None,
        baseline: Optional[Dict[str, Any]] = None,
        focus_run_ids: Optional[List[Any]] = None,
        top_k_anomalous: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate all pipeline figures.

        Parameters
        ----------
        runs, columns : Layer 0 output.
        pooled_correlations, causal_aggregated, clusters : Layer 2 output
            (compute_pooled_correlations, analyze_per_run, cluster_columns).
        run_summary, baseline : Layer 3 output (drift_results["run_summary"],
            drift_results["baseline"] — or the equivalent reloaded from
            layer3/drift_summary.json).
        focus_run_ids : which runs get full per-run detail (propagation
            timeline, drift breakdown, statistical bar, variable-with-onset
            traces). Defaults to the top_k_anomalous runs by anomaly_score.

        Returns
        -------
        dict
            {figure_name: filepath or {column: filepath}}.
        """
        if corr_methods is None:
            corr_methods = ["pearson"]

        figures: Dict[str, Any] = {}

        if runs and columns:
            var_overlays = self.plot_per_variable_overlays(runs, columns)
            if var_overlays:
                figures["var_overlays"] = var_overlays
            var_histograms = self.plot_per_variable_histograms(runs, columns)
            if var_histograms:
                figures["var_histograms"] = var_histograms

        if pooled_correlations:
            for method in corr_methods:
                corr_df = pooled_correlations.get(method)
                if corr_df is None:
                    continue
                path = self.plot_correlation_heatmap(corr_df, method=method)
                if path:
                    figures[f"heatmap_{method}"] = path
                path = self.plot_correlation_clustermap(corr_df, clusters, method=method)
                if path:
                    figures[f"clustermap_{method}"] = path

        if causal_aggregated:
            path = self.plot_causal_frequency(causal_aggregated, top_n=20)
            if path:
                figures["causal_frequency"] = path
            path = self.plot_causal_graph(causal_aggregated)
            if path:
                figures["causal_graph"] = path
            path = self.plot_causal_time_series_graph(causal_aggregated)
            if path:
                figures["causal_time_series"] = path
            path = self.plot_causal_lag_profile(causal_aggregated, top_n=10)
            if path:
                figures["causal_lag_profiles"] = path

        if run_summary:
            path = self.plot_anomaly_ranking(run_summary, top_n=top_k_anomalous)
            if path:
                figures["anomaly_ranking"] = path

            ids = focus_run_ids
            if ids is None:
                ranked = sorted(run_summary.items(), key=lambda kv: kv[1].get("anomaly_score", 0), reverse=True)
                ids = [rid for rid, _ in ranked[:top_k_anomalous]]

            path = self.plot_run_drift_breakdown(run_summary, ids)
            if path:
                figures["run_drift_breakdown"] = path

            for rid in ids:
                path = self.plot_cusum_propagation(run_summary, rid)
                if path:
                    figures[f"cusum_propagation_{rid}"] = path
                path = self.plot_statistical_drift_bar(run_summary, rid)
                if path:
                    figures[f"statistical_drift_{rid}"] = path
                if runs and baseline:
                    path = self.plot_variable_with_onset(runs, rid, baseline, run_summary)
                    if path:
                        figures[f"variable_onset_{rid}"] = path

        return figures

    # ═══════════════════════════════════════════════════════════════════
    # Standalone: load from a saved analysis_output/<dataset_name> folder
    # ═══════════════════════════════════════════════════════════════════

    @classmethod
    def load_from_folder(cls, dataset_dir: str) -> dict:
        """
        Load everything plotting needs from a saved dataset folder (see
        utils/results_exporter.py for the exact layout). Never touches
        Layer 0-3 — this is a pure reload.

        Returns
        -------
        dict with keys:
            runs, columns, pooled_correlations, clusters, causal_aggregated,
            run_summary, baseline
        """
        columns = None
        cols_path = os.path.join(dataset_dir, "run_data", "columns.json")
        if os.path.exists(cols_path):
            with open(cols_path) as f:
                columns = json.load(f)
        else:
            meta_path = os.path.join(dataset_dir, "dataset_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    columns = json.load(f).get("columns", [])

        if not columns:
            raise FileNotFoundError(
                f"No column info found in {dataset_dir}/. "
                f"Need run_data/columns.json or dataset_meta.json."
            )

        runs = {}
        run_data_dir = os.path.join(dataset_dir, "run_data")
        if os.path.isdir(run_data_dir):
            for fname in sorted(os.listdir(run_data_dir)):
                if fname.startswith("run_") and fname.endswith(".csv"):
                    run_id = fname[len("run_"):-len(".csv")]
                    try:
                        run_id = int(run_id)
                    except ValueError:
                        pass
                    runs[run_id] = pd.read_csv(os.path.join(run_data_dir, fname), index_col=0)

        pooled_correlations = {}
        corr_dir = os.path.join(dataset_dir, "layer2")
        if os.path.isdir(corr_dir):
            for fname in os.listdir(corr_dir):
                if fname.startswith("correlations_") and fname.endswith(".csv"):
                    method = fname[len("correlations_"):-len(".csv")]
                    pooled_correlations[method] = pd.read_csv(
                        os.path.join(corr_dir, fname), index_col=0
                    )

        causal_aggregated = None
        causal_path = os.path.join(dataset_dir, "layer2", "causal_per_run.json")
        if os.path.exists(causal_path):
            with open(causal_path) as f:
                causal_aggregated = json.load(f)

        clusters = None
        clusters_path = os.path.join(dataset_dir, "layer2", "clusters.json")
        if os.path.exists(clusters_path):
            with open(clusters_path) as f:
                clusters = json.load(f)

        run_summary, baseline = None, None
        drift_path = os.path.join(dataset_dir, "layer3", "drift_summary.json")
        if os.path.exists(drift_path):
            with open(drift_path) as f:
                drift_summary = json.load(f)
            run_summary = drift_summary.get("run_summary")
            baseline = drift_summary.get("baseline")

        return {
            "runs": runs,
            "columns": columns,
            "pooled_correlations": pooled_correlations,
            "clusters": clusters,
            "causal_aggregated": causal_aggregated,
            "run_summary": run_summary,
            "baseline": baseline,
        }

    @classmethod
    def run_standalone(cls, dataset_dir: str, corr_methods: Optional[list] = None) -> dict:
        """
        Regenerate all figures from a saved analysis_output/<dataset_name>
        folder — no re-analysis needed.

        Usage:
            PipelinePlotter.run_standalone("analysis_output/te_process_normal")
        """
        if corr_methods is None:
            corr_methods = ["pearson"]

        print(f"Loading data from: {dataset_dir}/")
        data = cls.load_from_folder(dataset_dir)

        print(f"  Runs: {sorted(data['runs'].keys(), key=str)}")
        print(f"  Columns: {len(data['columns'])}")
        print(f"  Correlation methods: {list(data['pooled_correlations'].keys())}")
        print(f"  Clusters: {'yes' if data['clusters'] else 'no'}")
        print(f"  Causal data: {'yes' if data['causal_aggregated'] else 'no'}")
        print(f"  Drift/anomaly data: {'yes' if data['run_summary'] else 'no'}")

        figures_dir = os.path.join(dataset_dir, "figures")
        plotter = cls(figures_dir)

        print(f"\nGenerating figures to: {figures_dir}/")
        figures = plotter.generate_all(
            runs=data["runs"],
            columns=data["columns"],
            pooled_correlations=data["pooled_correlations"],
            clusters=data["clusters"],
            causal_aggregated=data["causal_aggregated"],
            run_summary=data["run_summary"],
            baseline=data["baseline"],
            corr_methods=corr_methods,
        )

        for name, val in figures.items():
            if isinstance(val, dict):
                total_size = sum(os.path.getsize(p) for p in val.values())
                print(f"  [OK] {name}: {len(val)} files ({total_size:,} bytes total)")
            else:
                print(f"  [OK] {name}: {os.path.basename(val)} ({os.path.getsize(val):,} bytes)")

        print(f"\nDone — {len(figures)} figure groups generated.")
        return figures


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m tpm.utils.plotting analysis_output/<dataset_name>")
        print()
        print("Regenerates all figures from a saved dataset folder — never")
        print("re-runs Layer 0-3. The folder should contain whichever of")
        print("run_data/, layer2/, layer3/ a previous pipeline run wrote;")
        print("figures are generated for whatever is present.")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    if not os.path.isdir(dataset_dir):
        print(f"Error: {dataset_dir} is not a directory")
        sys.exit(1)

    PipelinePlotter.run_standalone(dataset_dir)
