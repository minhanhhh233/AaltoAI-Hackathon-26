"""
Layer 2: Relational / Causal Analysis
--------------------------------------
Thin wrapper around Tigramite PCMCI for causal discovery.
All causal analysis results come directly from Tigramite — no custom
calculations added. Correlations come from pandas (pooled across runs).

Two modes:
    1. analyze_per_run() — runs PCMCI on each run independently, then
       aggregates link frequencies across runs (recommended for multi-run)
    2. analyze() — concatenates runs with boundary masking (legacy)

Pooled correlations:
    compute_pooled_correlations() — Pearson/Spearman across all runs
    using pandas .corr() on concatenated data.

Column clustering:
    cluster_columns() — groups columns into related clusters using
    hierarchical clustering on a correlation-derived distance. Domain
    -agnostic: works on any numeric tabular data (not just time series
    or sensor readings), since it only consumes a correlation matrix.

Interface contract:
    analyzer = CausalAnalyzer()

    # Per-run causal analysis (recommended)
    aggregated = analyzer.analyze_per_run(
        runs={1: df1, 2: df2, ...},
        columns=["col_a", "col_b", ...],
    )
    aggregated["link_frequency"]  → {"src|tgt|tau": count}
    aggregated["var_names"]       → list[str]
    aggregated["n_runs"]          → int

    # Pooled correlations
    corr_dict = analyzer.compute_pooled_correlations(
        runs, columns, methods=["pearson", "spearman"]
    )
    corr_dict["pearson"]          → pd.DataFrame

    # Column clustering (from the correlations above)
    clusters = analyzer.cluster_columns(corr_dict, method="spearman")
    clusters["clusters"]          → {cluster_id: [col, ...]}
    clusters["column_cluster"]    → {col: cluster_id}

THRESHOLD REGISTRY:
    +------------------+---------+----------+------------------------------+
    | Parameter        | Default | Grounded | Basis                        |
    +------------------+---------+----------+------------------------------+
    | tau_max          | 10      | N/A      | Search range for lags        |
    | pc_alpha         | 0.01    | Yes      | Fisher sig. for PC step      |
    | alpha_level      | 0.01    | Yes      | Fisher sig. for MCI step     |
    | test_type        | ParCorr | Yes      | Partial correlation test     |
    | n_clusters (k)   | auto    | Yes      | Selected by maximizing mean  |
    |                  |         |          | silhouette score over k, not |
    |                  |         |          | fixed/hand-picked            |
    +------------------+---------+----------+------------------------------+
"""

import warnings
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")

# Tigramite imports
try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
except ImportError:
    raise ImportError(
        "Tigramite is required for Layer 2. "
        "Install with: pip install tigramite"
    )

try:
    from sklearn.metrics import silhouette_score
except ImportError:
    raise ImportError(
        "scikit-learn is required for Layer 2 column clustering. "
        "Install with: pip install scikit-learn"
    )


class CausalAnalyzer:
    """
    Layer 2 causal discovery engine — thin wrapper around Tigramite PCMCI.

    All causal analysis output comes directly from Tigramite; no custom
    causal-discovery math is added. Correlation and clustering are
    implemented directly in this module (pandas / scipy / scikit-learn).

    Usage:
        analyzer = CausalAnalyzer(tau_max=10, pc_alpha=0.01)

        # Per-run analysis with frequency aggregation (recommended)
        aggregated = analyzer.analyze_per_run(
            runs={1: df1, 2: df2},
            columns=cols,
        )

        # Pooled correlations (for heatmap/dendrogram)
        corrs = analyzer.compute_pooled_correlations(
            runs, cols, methods=["pearson", "spearman"]
        )

        # Column clustering
        clusters = analyzer.cluster_columns(corrs, method="spearman")
    """

    def __init__(
        self,
        tau_max: int = 10,
        pc_alpha: float = 0.01,
        alpha_level: float = 0.01,
        test_type: str = "parcorr",
    ):
        """
        Parameters
        ----------
        tau_max : int
            Maximum time lag to test for causal links.
        pc_alpha : float
            Significance level for the PC condition-selection step.
        alpha_level : float
            Significance level for the MCI step (final link selection).
        test_type : str
            Independence test type. Currently only "parcorr".
        """
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.alpha_level = alpha_level
        self.test_type = test_type

        # Tigramite objects (populated after analyze())
        self.pcmci: Optional[PCMCI] = None
        self.dataframe: Optional[pp.DataFrame] = None
        self._last_results: Optional[Dict[str, Any]] = None

    # ═══════════════════════════════════════════════════════════════════
    # Pooled correlations (pandas)
    # ═══════════════════════════════════════════════════════════════════

    def compute_pooled_correlations(
        self,
        runs: Dict[Any, pd.DataFrame],
        columns: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute correlations pooled across all runs using pandas.

        Concatenates all runs and computes correlation on the combined
        data. This gives the overall correlation structure rather than
        per-run correlations.

        Parameters
        ----------
        runs : dict
            {run_id: DataFrame}
        columns : list of str, optional
            Columns to use. If None, uses all common numeric columns.
        methods : list of str
            Correlation methods. Default: ["pearson", "spearman"].

        Returns
        -------
        dict
            {method: pd.DataFrame} correlation matrices.
        """
        if methods is None:
            methods = ["pearson", "spearman"]

        run_ids = sorted(runs.keys())

        # Determine columns
        if columns is None:
            common = set(runs[run_ids[0]].columns)
            for rid in run_ids[1:]:
                common &= set(runs[rid].columns)
            columns = sorted(common)

        # Concatenate all runs
        frames = []
        for rid in run_ids:
            df = runs[rid][columns].select_dtypes(include=[np.number])
            frames.append(df)

        pooled = pd.concat(frames, ignore_index=True)

        # Drop zero-variance columns
        std = pooled.std()
        valid_cols = std[std > 1e-12].index.tolist()
        pooled = pooled[valid_cols]

        # Compute correlations
        result = {}
        for method in methods:
            result[method] = pooled.corr(method=method)

        return result

    # ═══════════════════════════════════════════════════════════════════
    # Column clustering (correlation-based)
    # ═══════════════════════════════════════════════════════════════════

    def cluster_columns(
        self,
        correlations: Dict[str, pd.DataFrame],
        method: str = "spearman",
        linkage_method: str = "average",
        max_clusters: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Group columns into clusters of mutually related variables using
        hierarchical clustering on a correlation-derived distance.

        Domain-agnostic by design: this only consumes a correlation
        matrix (the output of compute_pooled_correlations), so it
        applies to any numeric tabular dataset — sensor readings,
        financial series, survey responses, gene expression, etc.
        Nothing here assumes time order, a particular column count, or
        column naming.

        Distance is the standard correlation-to-distance transform for
        standardized variables:

            d_ij = sqrt(2 * (1 - r_ij))

        This is a true metric (unlike ad hoc "1 - |r|" distances):
        for unit-variance x, y, ||x - y||^2 = 2n(1 - r), so this is
        just that relationship rescaled. It's the same transform used
        for correlation-based clustering across many domains (e.g.
        Mantegna's hierarchical clustering of asset return correlations).

        The number of clusters is chosen automatically by maximizing
        the mean silhouette score over candidate cluster counts — a
        standard, data-driven model-selection criterion — rather than
        a fixed or hand-picked number.

        Parameters
        ----------
        correlations : dict
            Output of compute_pooled_correlations(), e.g.
            {"pearson": pd.DataFrame, "spearman": pd.DataFrame}.
        method : str
            Which correlation matrix in `correlations` to cluster on.
            Spearman (rank correlation) is the default since it only
            assumes a monotonic relationship, not linearity — more
            robust across different kinds of data than Pearson.
        linkage_method : str
            scipy linkage method: "average" (UPGMA, default), "complete",
            "ward", "single", etc.
        max_clusters : int, optional
            Largest cluster count to consider. Defaults to
            min(10, n_columns - 1).

        Returns
        -------
        dict
            - "clusters": {cluster_id: [col, ...]}
            - "column_cluster": {col: cluster_id}
            - "linkage_matrix": scipy linkage array (for dendrograms)
            - "columns": list[str] — columns actually clustered
            - "dropped_columns": list[str] — columns excluded because
              their correlation was undefined (e.g. zero-variance)
            - "distance_metric": "sqrt(2*(1-r))"
            - "correlation_method": which matrix was used
            - "silhouette_scores": {k: score} for every k evaluated
            - "best_k": chosen number of clusters
            - "best_silhouette": silhouette score at best_k
        """
        if method not in correlations:
            raise ValueError(
                f"'{method}' not found in correlations dict. "
                f"Available: {list(correlations.keys())}"
            )

        corr = correlations[method].copy()

        # Drop columns with any undefined correlation (e.g. zero-variance
        # columns produce NaN rows/cols) — clustering needs a complete
        # distance matrix.
        valid_mask = corr.notna().all(axis=0) & corr.notna().all(axis=1)
        dropped_columns = corr.columns[~valid_mask].tolist()
        corr = corr.loc[valid_mask, valid_mask]
        columns = corr.columns.tolist()
        n = len(columns)

        if n < 3:
            return {
                "status": "skipped",
                "reason": "fewer than 3 valid columns to cluster",
                "columns": columns,
                "dropped_columns": dropped_columns,
            }

        # Correlation -> proper metric distance
        r = np.clip(corr.values, -1.0, 1.0)
        np.fill_diagonal(r, 1.0)
        dist = np.sqrt(np.clip(2 * (1 - r), 0, None))
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0  # kill float asymmetry before squareform

        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method=linkage_method)

        if max_clusters is None:
            max_clusters = min(10, n - 1)
        max_clusters = max(2, max_clusters)

        best_k, best_score, scores = None, -1.0, {}
        for k in range(2, max_clusters + 1):
            labels = fcluster(Z, t=k, criterion="maxclust")
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(dist, labels, metric="precomputed")
            scores[k] = round(float(score), 4)
            if score > best_score:
                best_score = score
                best_k = k

        if best_k is None:
            # Every cut degenerated to a single cluster — report one.
            best_k = 1
            labels = np.ones(n, dtype=int)
            best_score = None
        else:
            labels = fcluster(Z, t=best_k, criterion="maxclust")

        clusters = defaultdict(list)
        column_cluster = {}
        for col, lbl in zip(columns, labels):
            clusters[int(lbl)].append(col)
            column_cluster[col] = int(lbl)

        return {
            "status": "ok",
            "clusters": dict(clusters),
            "column_cluster": column_cluster,
            "linkage_matrix": Z,
            "columns": columns,
            "dropped_columns": dropped_columns,
            "distance_metric": "sqrt(2*(1-r))",
            "correlation_method": method,
            "linkage_method": linkage_method,
            "silhouette_scores": scores,
            "best_k": best_k,
            "best_silhouette": (
                round(float(best_score), 4) if best_score is not None else None
            ),
        }

    # ═══════════════════════════════════════════════════════════════════
    # Per-run causal analysis with frequency aggregation
    # ═══════════════════════════════════════════════════════════════════

    def analyze_per_run(
        self,
        runs: Dict[Any, pd.DataFrame],
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run PCMCI on each run independently and aggregate results.

        For each run, runs PCMCI and collects significant links.
        Then aggregates: for each link (src, tgt, tau), counts how
        many runs it was detected in.

        Parameters
        ----------
        runs : dict
            {run_id: DataFrame} — each an independent time series.
        columns : list of str, optional
            Columns to analyze. If None, uses all common numeric columns.

        Returns
        -------
        dict
            - "link_frequency": {"src|tgt|tau": count}
            - "mean_val": {"src|tgt|tau": mean_partial_corr}
            - "mean_pval": {"src|tgt|tau": mean_pvalue}
            - "lag_profiles": {"src|tgt": {taus, mean_vals, std_vals,
              frequencies}} for each variable pair
            - "per_run_results": [{run_id, n_significant, ...}]
            - "var_names": list[str]
            - "n_runs": int
            - "metadata": dict
        """
        t_start = time.time()
        run_ids = sorted(runs.keys())
        n_runs = len(run_ids)

        # ── Determine columns ────────────────────────────────────
        if columns is None:
            common = set(runs[run_ids[0]].columns)
            for rid in run_ids[1:]:
                common &= set(runs[rid].columns)
            columns = sorted(common)

        # Filter to numeric, non-zero-variance across all runs
        valid_cols = []
        for col in columns:
            vals = []
            for rid in run_ids:
                if col not in runs[rid].columns:
                    break
                df_col = pd.to_numeric(runs[rid][col], errors="coerce")
                vals.extend(df_col.dropna().tolist())
            else:
                if len(vals) > 5 and np.std(vals) > 1e-12:
                    valid_cols.append(col)

        columns = valid_cols
        n_vars = len(columns)

        # ── Run PCMCI per run ────────────────────────────────────
        # Accumulators
        link_counts = defaultdict(int)        # "src|tgt|tau" -> count
        link_vals = defaultdict(list)         # "src|tgt|tau" -> [val, ...]
        link_pvals = defaultdict(list)        # "src|tgt|tau" -> [pval, ...]
        per_run_info = []

        for rid in run_ids:
            df = runs[rid][columns].copy()

            # Convert to numeric, fill NaN with column mean
            data = df.select_dtypes(include=[np.number]).values.astype(
                np.float64
            )
            for j in range(data.shape[1]):
                col_data = data[:, j]
                nan_mask = np.isnan(col_data)
                if nan_mask.any():
                    col_data[nan_mask] = np.nanmean(col_data)
                    data[:, j] = col_data

            if len(data) < self.tau_max + 5:
                per_run_info.append({
                    "run_id": str(rid),
                    "status": "skipped",
                    "reason": "too_few_samples",
                    "n_samples": len(data),
                })
                continue

            # Build Tigramite DataFrame
            tg_df = pp.DataFrame(data, var_names=columns)
            cond_ind_test = ParCorr(significance="analytic")

            pcmci = PCMCI(
                dataframe=tg_df,
                cond_ind_test=cond_ind_test,
                verbosity=0,
            )

            results = pcmci.run_pcmci(
                tau_max=self.tau_max,
                pc_alpha=self.pc_alpha,
            )

            # Extract significant links
            p_matrix = results["p_matrix"]
            val_matrix = results["val_matrix"]

            n_sig = 0
            for j in range(n_vars):
                for i in range(n_vars):
                    for tau in range(0, self.tau_max + 1):
                        if i == j and tau == 0:
                            continue
                        if p_matrix[i, j, tau] < self.alpha_level:
                            key = f"{columns[i]}|{columns[j]}|{tau}"
                            link_counts[key] += 1
                            link_vals[key].append(
                                float(val_matrix[i, j, tau])
                            )
                            link_pvals[key].append(
                                float(p_matrix[i, j, tau])
                            )
                            n_sig += 1

            per_run_info.append({
                "run_id": str(rid),
                "status": "ok",
                "n_samples": len(data),
                "n_significant": n_sig,
            })

            # Keep last PCMCI for reference
            self.pcmci = pcmci
            self.dataframe = tg_df

        # ── Aggregate results ────────────────────────────────────
        mean_val = {}
        mean_pval = {}
        for key in link_counts:
            mean_val[key] = float(np.mean(link_vals[key]))
            mean_pval[key] = float(np.mean(link_pvals[key]))

        # Build lag profiles for each variable pair
        lag_profiles = self._build_lag_profiles(
            link_counts, link_vals, n_runs, columns,
        )

        elapsed = time.time() - t_start
        n_completed = sum(
            1 for r in per_run_info if r.get("status") == "ok"
        )

        result = {
            "link_frequency": dict(link_counts),
            "mean_val": mean_val,
            "mean_pval": mean_pval,
            "lag_profiles": lag_profiles,
            "per_run_results": per_run_info,
            "var_names": columns,
            "n_runs": n_completed,
            "n_total_runs": n_runs,
            "metadata": {
                "n_variables": n_vars,
                "n_runs": n_completed,
                "n_total_runs": n_runs,
                "run_ids": [str(r) for r in run_ids],
                "elapsed_seconds": round(elapsed, 2),
                "parameters": {
                    "tau_max": self.tau_max,
                    "pc_alpha": self.pc_alpha,
                    "alpha_level": self.alpha_level,
                    "test_type": self.test_type,
                },
            },
        }

        self._last_results = result
        return result

    def _build_lag_profiles(
        self,
        link_counts: Dict[str, int],
        link_vals: Dict[str, list],
        n_runs: int,
        var_names: List[str],
    ) -> Dict[str, Dict]:
        """
        Build lag profiles for each variable pair.

        For each pair (src, tgt), collects frequency and mean value
        at each lag.
        """
        # Group by (src, tgt) pair
        pair_data = defaultdict(lambda: defaultdict(lambda: {
            "count": 0, "vals": [],
        }))

        for key, count in link_counts.items():
            src, tgt, tau_str = key.split("|")
            tau = int(tau_str)
            pair_key = f"{src}|{tgt}"
            pair_data[pair_key][tau]["count"] = count
            pair_data[pair_key][tau]["vals"] = link_vals[key]

        profiles = {}
        for pair_key, tau_dict in pair_data.items():
            taus = sorted(tau_dict.keys())
            if not taus:
                continue
            frequencies = [tau_dict[t]["count"] for t in taus]
            all_vals = [tau_dict[t]["vals"] for t in taus]
            mean_vals = [float(np.mean(np.abs(v))) if v else 0.0
                         for v in all_vals]
            std_vals = [float(np.std(np.abs(v))) if len(v) > 1 else 0.0
                        for v in all_vals]

            profiles[pair_key] = {
                "taus": taus,
                "frequencies": frequencies,
                "mean_vals": mean_vals,
                "std_vals": std_vals,
            }

        return profiles

    # ═══════════════════════════════════════════════════════════════════
    # Legacy: concatenated multi-run analysis
    # ═══════════════════════════════════════════════════════════════════

    def analyze(
        self,
        runs: Dict[Any, pd.DataFrame],
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run PCMCI causal discovery on multi-run data (legacy).

        Concatenates runs with boundary masking. For new pipelines,
        prefer analyze_per_run() which gives link frequency information.

        Parameters
        ----------
        runs : dict
            {run_id: DataFrame}
        columns : list of str, optional
            Columns to analyze.

        Returns
        -------
        dict
            Raw Tigramite output plus metadata.
        """
        t_start = time.time()
        run_ids = sorted(runs.keys())

        # ── Determine columns ────────────────────────────────────
        if columns is None:
            common = set(runs[run_ids[0]].columns)
            for rid in run_ids[1:]:
                common &= set(runs[rid].columns)
            columns = sorted(common)

        # Filter to numeric only and drop zero-variance
        dfs = []
        for rid in run_ids:
            df = runs[rid][columns].select_dtypes(include=[np.number])
            dfs.append(df)

        valid_cols = []
        for col in columns:
            if col not in dfs[0].columns:
                continue
            all_vals = pd.concat([df[col] for df in dfs]).dropna()
            if len(all_vals) > 5 and all_vals.std() > 1e-12:
                valid_cols.append(col)

        columns = valid_cols
        n_vars = len(columns)

        # ── Concatenate runs with boundary masking ───────────────
        data_arrays = []
        run_lengths = []

        for df in dfs:
            arr = df[columns].values.astype(np.float64)
            data_arrays.append(arr)
            run_lengths.append(len(arr))

        all_data = np.vstack(data_arrays)
        total_T = all_data.shape[0]

        mask = np.zeros(all_data.shape, dtype=bool)
        offset = 0
        for i, length in enumerate(run_lengths):
            if i > 0:
                mask[offset:offset + self.tau_max, :] = True
            offset += length

        for j in range(n_vars):
            col_data = all_data[:, j]
            nan_mask = np.isnan(col_data)
            if nan_mask.any():
                col_mean = np.nanmean(col_data)
                col_data[nan_mask] = col_mean
                mask[nan_mask, j] = True
                all_data[:, j] = col_data

        # ── Run PCMCI ────────────────────────────────────────────
        self.dataframe = pp.DataFrame(
            all_data,
            var_names=columns,
            mask=mask,
        )

        # mask_type="xyz" is required for the boundary mask above to have
        # any effect — without it, Tigramite ignores dataframe.mask entirely
        # and treats the concatenated runs as one continuous series.
        cond_ind_test = ParCorr(significance="analytic", mask_type="xyz")

        self.pcmci = PCMCI(
            dataframe=self.dataframe,
            cond_ind_test=cond_ind_test,
            verbosity=0,
        )

        results = self.pcmci.run_pcmci(
            tau_max=self.tau_max,
            pc_alpha=self.pc_alpha,
        )

        elapsed = time.time() - t_start
        results["var_names"] = columns
        results["metadata"] = {
            "n_variables": n_vars,
            "n_runs": len(run_ids),
            "run_ids": [str(r) for r in run_ids],
            "total_samples": total_T,
            "run_lengths": run_lengths,
            "masked_samples": int(mask.sum(axis=0)[0]),
            "elapsed_seconds": round(elapsed, 2),
            "parameters": {
                "tau_max": self.tau_max,
                "pc_alpha": self.pc_alpha,
                "alpha_level": self.alpha_level,
                "test_type": self.test_type,
            },
        }

        self._last_results = results
        return results

    @property
    def results(self) -> Optional[Dict[str, Any]]:
        """Access the last analysis results."""
        return self._last_results


# Keep backward-compatible name
RelationalAnalyzer = CausalAnalyzer