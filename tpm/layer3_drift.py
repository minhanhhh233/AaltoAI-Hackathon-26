"""
Layer 3: Drift and Anomaly Detection
-------------------------------------
Builds a baseline from pooled run statistics, then scores each individual
run for deviation across four perspectives:

    1. Statistical drift   — per-variable z-score of run mean shift
                             and F-test of variance ratio
    2. Correlation drift   — per-run correlation matrix vs pooled,
                             measured by Frobenius norm + Fisher-z
                             per-element tests
    3. Causal drift        — per-run causal links vs consensus links
                             (links detected in ALL runs)
    4. Onset detection     — per-variable, per-sample one-sided CUSUM
                             control chart (Page, 1954) against the
                             baseline mean/std, localizing the first
                             sample where a sustained shift begins.
                             Ordering variables by onset sample gives
                             a first-pass propagation sequence.

All thresholds come from statistical formulas (z-critical values,
F-distribution quantiles, Fisher z-transformation, CUSUM ARL0
approximation). The LLM does NOT set thresholds.

Before any of the above runs, every scored batch is gated on data
quality (utils/data_quality.py — shared with Layer 1): a missing-values
check, plus any operator-authored custom rules for this dataset
(utils/quality_rules.py — taught through the dashboard chat and
confirmed by a human before being saved). A run whose quality verdict
is "critical" is EXCLUDED from the four perspectives above rather than
producing a possibly-meaningless drift/anomaly score from untrustworthy
numbers (e.g. a z-test on a mostly-missing column can be numerically
well-defined and still semantically garbage) — its run_summary entry
states the gate failure and the specific reasons explicitly, rather
than silently reporting an anomaly_score that looks like a normal
"not drifted" result.

Perspectives 1-2 correct for autocorrelation: row-ordered data (time
series, or any other sequentially-dependent data) violates the i.i.d.
assumption a plain z/F/Fisher-z test makes, which otherwise makes
these tests badly over-sensitive. Raw sample counts are replaced with
an autocorrelation-corrected effective sample size (ESS), estimated
directly from each column's own lag-1 autocorrelation — grounded
(Zwiers & von Storch, 1995 for univariate ESS; Pyper & Peterman, 1998
for the pairwise/correlation extension), automatic, and generic to any
sequentially-ordered numeric column.

Interface contract:
    detector = DriftDetector()

    # Build baseline from all runs
    detector.build_baseline(
        runs={1: df1, 2: df2, ...},
        columns=["col_a", "col_b", ...],
        pooled_correlations={"pearson": pd.DataFrame},  # from Layer 2
        causal_aggregated={...},                          # from Layer 2
    )

    # Score each run against baseline
    results = detector.score_runs()
    results["statistical"]       → per-run, per-variable drift scores
    results["correlation"]       → per-run correlation drift
    results["causal"]            → per-run causal link drift
    results["cusum"]             → per-run, per-variable onset sample
    results["run_summary"]       → overall anomaly score per run

THRESHOLD REGISTRY:
    +---------------------+------------+----------+------------------------------+
    | Parameter           | Default    | Grounded | Basis                        |
    +---------------------+------------+----------+------------------------------+
    | stat_alpha          | 0.01       | Yes      | Two-sided z-test (Bonferroni),|
    |                     |            |          | uses autocorrelation-         |
    |                     |            |          | corrected effective n         |
    | var_alpha           | 0.01       | Yes      | F-test for equal variances,  |
    |                     |            |          | df from effective n           |
    | corr_alpha          | 0.01       | Yes      | Fisher z-transform for       |
    |                     |            |          | correlation comparison, uses  |
    |                     |            |          | pairwise effective n           |
    | consensus_threshold | 1.0        | Yes      | Fraction of runs for a link  |
    |                     |            |          | to be "consensus" (1.0=all)  |
    | cusum_shift_sigma   | 1.0        | Yes      | Min. mean shift (in baseline |
    |                     |            |          | std-devs) CUSUM is tuned to  |
    |                     |            |          | detect; sets reference value |
    |                     |            |          | k = cusum_shift_sigma / 2    |
    | cusum_arl0          | 5000.0     | Yes      | Target FAMILY-WISE in-control|
    |                     |            |          | average run length (across   |
    |                     |            |          | all variables, both          |
    |                     |            |          | directions); per-channel h   |
    |                     |            |          | solved from cusum_arl0 * 2 * |
    |                     |            |          | n_vars via Siegmund's ARL    |
    |                     |            |          | approximation (Bonferroni-   |
    |                     |            |          | style correction, same idea  |
    |                     |            |          | as stat_alpha/var_alpha)     |
    +---------------------+------------+----------+------------------------------+
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from collections import defaultdict
from scipy import stats
from scipy.optimize import brentq

from tpm.utils.data_quality import check_data_quality


class DriftDetector:
    """
    Layer 3: Drift and Anomaly Detection.

    Builds a baseline from pooled statistics across all runs, then
    scores each run against it using four complementary perspectives.

    Usage:
        detector = DriftDetector()
        detector.build_baseline(runs, columns,
                                pooled_correlations, causal_aggregated)
        results = detector.score_runs()
    """

    def __init__(
        self,
        stat_alpha: float = 0.01,
        var_alpha: float = 0.01,
        corr_alpha: float = 0.01,
        consensus_threshold: float = 1.0,
        cusum_shift_sigma: float = 1.0,
        cusum_arl0: float = 5000.0,
    ):
        """
        Parameters
        ----------
        stat_alpha : float
            Significance level for mean-shift z-test.
            Bonferroni-corrected by n_variables internally.
        var_alpha : float
            Significance level for F-test of variance ratio.
            Bonferroni-corrected by n_variables internally.
        corr_alpha : float
            Significance level for Fisher-z correlation comparison.
        consensus_threshold : float
            Fraction of runs a causal link must appear in to be
            "consensus". 1.0 = must appear in ALL runs.
        cusum_shift_sigma : float
            Minimum sustained mean shift (in baseline standard
            deviations) the CUSUM onset chart is tuned to detect.
            Sets the CUSUM reference value k = cusum_shift_sigma / 2
            (the standard SPC choice — optimal for shifts of this
            size). Default 1.0 = tuned to detect a 1-sigma shift.
        cusum_arl0 : float
            Target *family-wise* in-control average run length: the
            expected number of in-control samples before a false
            onset alarm fires on ANY variable, in EITHER direction.
            Each variable runs two one-sided charts (increase/
            decrease), so — analogous to the Bonferroni correction
            applied to stat_alpha/var_alpha — the per-channel decision
            interval h is solved from an inflated per-channel target
            of cusum_arl0 * 2 * n_variables, computed fresh in
            _score_cusum() once n_variables is known. Without this
            correction, false-alarm probability compounds across
            variables and directions. Larger cusum_arl0 = fewer false
            alarms but slower detection.
        """
        self.stat_alpha = stat_alpha
        self.var_alpha = var_alpha
        self.corr_alpha = corr_alpha
        self.consensus_threshold = consensus_threshold
        self.cusum_shift_sigma = cusum_shift_sigma
        self.cusum_arl0 = cusum_arl0
        self.cusum_k = cusum_shift_sigma / 2.0
        # h depends on n_variables (multiple-comparisons correction),
        # so it is solved fresh in _score_cusum() once columns is known,
        # not here.

        # Populated by build_baseline()
        self._runs: Dict[Any, pd.DataFrame] = {}
        self._columns: List[str] = []
        self._run_ids: List[Any] = []
        self._baseline: Dict[str, Any] = {}
        self._pooled_corr: Optional[pd.DataFrame] = None
        self._causal_aggregated: Optional[Dict[str, Any]] = None

    # ═══════════════════════════════════════════════════════════════════
    # Build baseline
    # ═══════════════════════════════════════════════════════════════════

    def build_baseline(
        self,
        runs: Dict[Any, pd.DataFrame],
        columns: List[str],
        pooled_correlations: Optional[Dict[str, pd.DataFrame]] = None,
        causal_aggregated: Optional[Dict[str, Any]] = None,
    ):
        """
        Build the baseline from all runs.

        Parameters
        ----------
        runs : dict
            {run_id: DataFrame} — each an independent time series.
        columns : list of str
            Column names to analyze.
        pooled_correlations : dict, optional
            {"pearson": pd.DataFrame, ...} from Layer 2.
        causal_aggregated : dict, optional
            Per-run causal results from Layer 2.
        """
        self._runs = runs
        self._run_ids = sorted(runs.keys())
        self._columns = columns

        # Store Layer 2 outputs
        if pooled_correlations and "pearson" in pooled_correlations:
            self._pooled_corr = pooled_correlations["pearson"]
        self._causal_aggregated = causal_aggregated

        # Compute per-variable pooled statistics
        baseline = {}
        for col in columns:
            # Pool values across all runs
            all_vals = []
            per_run_n = {}
            per_run_ess = {}
            per_run_rho1 = {}
            for rid in self._run_ids:
                col_series = runs[rid][col].dropna()
                col_data = col_series.values.astype(np.float64)
                all_vals.append(col_data)
                n_run = len(col_data)
                rho1 = self._lag1_autocorr(col_data)
                per_run_n[rid] = n_run
                per_run_rho1[rid] = rho1
                per_run_ess[rid] = self._effective_sample_size(n_run, rho1)

            pooled = np.concatenate(all_vals)
            total_n = sum(per_run_n.values())
            # Length-weighted average lag-1 autocorrelation, used later to
            # approximate the effective sample size behind a *pairwise*
            # pooled correlation (see _score_correlation).
            mean_rho1 = (
                sum(per_run_rho1[rid] * per_run_n[rid] for rid in self._run_ids)
                / total_n if total_n > 0 else 0.0
            )

            baseline[col] = {
                "pooled_mean": float(np.mean(pooled)),
                "pooled_std": float(np.std(pooled, ddof=1)),
                "pooled_n": len(pooled),
                "pooled_ess": float(sum(per_run_ess.values())),
                "mean_rho1": float(mean_rho1),
                "per_run_n": per_run_n,
            }

        self._baseline = baseline

    # ═══════════════════════════════════════════════════════════════════
    # Score all runs
    # ═══════════════════════════════════════════════════════════════════

    def score_runs(
        self,
        test_runs: Optional[Dict[Any, pd.DataFrame]] = None,
        custom_rules: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Score runs against the baseline.

        Parameters
        ----------
        test_runs : dict, optional
            {run_id: DataFrame} — runs to score. If None, scores the
            same runs used to build the baseline (self-check mode).
            Pass fault/new data here to detect drift against normal
            baseline.
        custom_rules : list of dict, optional
            Operator-authored data-quality rules for this dataset (see
            utils/quality_rules.py), typically loaded fresh from disk by
            the caller right before scoring so a rule taught since the
            baseline was built still applies.

        Returns
        -------
        dict with keys:
            "data_quality" : per-run data-quality report + trust verdict
                             (computed BEFORE any of the below; a run
                             with a "critical" verdict is excluded from
                             the four perspectives — see run_summary's
                             "data_quality_gate" for that run instead)
            "statistical"  : per-run, per-variable mean/variance drift
            "correlation"  : per-run correlation matrix drift
            "causal"       : per-run causal link drift
            "run_summary"  : overall anomaly summary per run
            "baseline"     : baseline statistics used
            "metadata"     : parameters and thresholds
        """
        # Determine which runs to score
        if test_runs is not None:
            score_runs = test_runs
            score_ids = sorted(test_runs.keys())
            mode = "test"
        else:
            score_runs = self._runs
            score_ids = self._run_ids
            mode = "self_check"

        # ── Data quality gate — before any drift/fault reasoning ────
        quality_results = self._score_data_quality(score_runs, score_ids, custom_rules)
        gated_ids = {
            rid for rid in score_ids
            if quality_results[str(rid)]["overall"]["severity"] == "critical"
        }
        analyzable_ids = [rid for rid in score_ids if rid not in gated_ids]

        stat_results = self._score_statistical(score_runs, analyzable_ids)
        corr_results = self._score_correlation(score_runs, analyzable_ids)
        causal_results = self._score_causal()
        cusum_results = self._score_cusum(score_runs, analyzable_ids)

        # Build per-run summary (over ALL score_ids, including gated ones —
        # _build_run_summary reports them with an explicit gate-failure
        # entry rather than silently omitting them)
        run_summary = self._build_run_summary(
            stat_results, corr_results, causal_results, cusum_results,
            score_ids, quality_results=quality_results,
        )

        n_vars = len(self._columns)
        n_score_runs = len(score_ids)

        return {
            "data_quality": quality_results,
            "statistical": stat_results,
            "correlation": corr_results,
            "causal": causal_results,
            "cusum": cusum_results,
            "run_summary": run_summary,
            "baseline": {
                col: {
                    "pooled_mean": round(v["pooled_mean"], 6),
                    "pooled_std": round(v["pooled_std"], 6),
                }
                for col, v in self._baseline.items()
            },
            "metadata": {
                "mode": mode,
                "n_baseline_runs": len(self._run_ids),
                "n_test_runs": n_score_runs,
                "n_variables": n_vars,
                "baseline_run_ids": [str(r) for r in self._run_ids],
                "test_run_ids": [str(r) for r in score_ids],
                "thresholds": {
                    "stat_alpha": self.stat_alpha,
                    "stat_alpha_bonferroni": self.stat_alpha / max(n_vars, 1),
                    "var_alpha": self.var_alpha,
                    "var_alpha_bonferroni": self.var_alpha / max(n_vars, 1),
                    "corr_alpha": self.corr_alpha,
                    "consensus_threshold": self.consensus_threshold,
                    "cusum_shift_sigma": self.cusum_shift_sigma,
                    "cusum_arl0_family_wise": self.cusum_arl0,
                    "cusum_effective_arl0_per_channel": cusum_results.get(
                        "effective_arl0_per_channel"
                    ),
                    "cusum_k": cusum_results.get("k"),
                    "cusum_h": cusum_results.get("h"),
                },
            },
        }

    # ═══════════════════════════════════════════════════════════════════
    # Effective sample size (autocorrelation correction)
    # ═══════════════════════════════════════════════════════════════════
    #
    # The z-test, F-test, and Fisher-z tests below implicitly assume
    # i.i.d. samples. Row-ordered data (time series, or any other
    # sequentially-dependent data) violates that: consecutive samples
    # are correlated, so the raw sample count overstates how much
    # independent information is actually present. Left uncorrected,
    # this makes every test in this perspective badly over-sensitive.
    # These helpers replace raw n with an effective sample size (ESS)
    # derived from the data's own lag-1 autocorrelation — grounded,
    # generic (works for any sequentially-ordered numeric column, not
    # just sensors), and not a hand-picked threshold.

    @staticmethod
    def _lag1_autocorr(x: np.ndarray) -> float:
        """
        Lag-1 autocorrelation of a 1-D array, used to estimate how
        much consecutive samples depend on each other. Returns 0.0
        (i.e. "assume independent, no correction") if there are too
        few samples or the series has no variance to measure.
        """
        n = len(x)
        if n < 4:
            return 0.0
        x0, x1 = x[:-1], x[1:]
        if np.std(x0) < 1e-12 or np.std(x1) < 1e-12:
            return 0.0
        rho = np.corrcoef(x0, x1)[0, 1]
        if np.isnan(rho):
            return 0.0
        return float(np.clip(rho, -0.999, 0.999))

    @staticmethod
    def _effective_sample_size(n: int, rho1: float) -> float:
        """
        Effective sample size for an autocorrelated series, via the
        standard AR(1) approximation (Zwiers & von Storch, 1995):

            n_eff = n * (1 - rho1) / (1 + rho1)

        Positive autocorrelation (the common case — consecutive
        samples resemble each other) shrinks n_eff below n; negative
        autocorrelation would inflate it, so the result is clipped to
        [2, n] to stay conservative. Falls back to n itself (no
        correction) when there are too few samples to estimate rho1
        reliably.
        """
        if n < 4:
            return float(max(n, 1))
        ess = n * (1 - rho1) / (1 + rho1)
        return float(np.clip(ess, 2.0, n))

    @staticmethod
    def _effective_sample_size_pair(n: int, rho1_x: float, rho1_y: float) -> float:
        """
        Effective sample size behind a Pearson/Spearman correlation
        between two autocorrelated series — the bivariate extension of
        the AR(1) approximation above (Pyper & Peterman, 1998):

            n_eff = n * (1 - rho1_x * rho1_y) / (1 + rho1_x * rho1_y)
        """
        if n < 4:
            return float(max(n, 1))
        prod = rho1_x * rho1_y
        denom = 1 + prod
        ess = n * (1 - prod) / denom if abs(denom) > 1e-9 else float(n)
        return float(np.clip(ess, 2.0, n))

    # ═══════════════════════════════════════════════════════════════════
    # Data quality gate (runs before any of the perspectives below)
    # ═══════════════════════════════════════════════════════════════════

    def _score_data_quality(
        self,
        score_runs: Dict[Any, pd.DataFrame],
        score_ids: List[Any],
        custom_rules: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run the shared data-quality checks (utils/data_quality.py,
        also used by Layer 1) on each run being scored. A "critical"
        verdict here is what excludes a run from the four perspectives
        in score_runs() — see that method and the module docstring for
        why.
        """
        return {
            str(rid): check_data_quality(score_runs[rid], self._columns, custom_rules=custom_rules)
            for rid in score_ids
        }

    # ═══════════════════════════════════════════════════════════════════
    # Perspective 1: Statistical drift (mean shift + variance ratio)
    # ═══════════════════════════════════════════════════════════════════

    def _score_statistical(
        self,
        score_runs: Dict[Any, pd.DataFrame],
        score_ids: List[Any],
    ) -> Dict[str, Any]:
        """
        For each run and each variable, test whether the run's mean
        and variance differ significantly from the pooled baseline.

        Mean shift: z-test with Bonferroni correction.
            z = (run_mean - pooled_mean) / (pooled_std / sqrt(n_eff_run))

        Variance ratio: F-test (two-sided) with Bonferroni correction.
            F = run_var / pooled_var

        n_run/pooled_n are replaced by autocorrelation-corrected
        effective sample sizes (see _effective_sample_size above) —
        row-ordered data violates the i.i.d. assumption these tests
        otherwise make, which would make them badly over-sensitive.
        """
        n_vars = len(self._columns)
        alpha_mean = self.stat_alpha / max(n_vars, 1)  # Bonferroni
        alpha_var = self.var_alpha / max(n_vars, 1)

        # z critical value (two-sided)
        z_crit = stats.norm.ppf(1 - alpha_mean / 2)

        per_run = {}

        for rid in score_ids:
            run_df = score_runs[rid]
            variables = {}
            n_drifted_mean = 0
            n_drifted_var = 0

            for col in self._columns:
                bl = self._baseline[col]
                pooled_mean = bl["pooled_mean"]
                pooled_std = bl["pooled_std"]
                pooled_ess = bl["pooled_ess"]

                col_data = run_df[col].dropna().values.astype(np.float64)
                n_run = len(col_data)
                run_mean = float(np.mean(col_data))
                run_std = float(np.std(col_data, ddof=1))

                rho1_run = self._lag1_autocorr(col_data)
                ess_run = self._effective_sample_size(n_run, rho1_run)

                # -- Mean shift z-test --
                if pooled_std > 1e-12 and ess_run > 1:
                    se = pooled_std / np.sqrt(ess_run)
                    z_score = (run_mean - pooled_mean) / se
                    p_mean = 2 * (1 - stats.norm.cdf(abs(z_score)))
                    mean_drifted = abs(z_score) > z_crit
                else:
                    z_score = 0.0
                    p_mean = 1.0
                    mean_drifted = False

                # -- Variance ratio F-test (two-sided) --
                pooled_var = pooled_std ** 2
                run_var = run_std ** 2

                if pooled_var > 1e-12 and ess_run > 2 and pooled_ess > 2:
                    f_stat = run_var / pooled_var
                    df1 = ess_run - 1
                    df2 = pooled_ess - 1

                    # Two-sided F-test
                    p_var_upper = 1 - stats.f.cdf(f_stat, df1, df2)
                    p_var_lower = stats.f.cdf(f_stat, df1, df2)
                    p_var = 2 * min(p_var_upper, p_var_lower)
                    var_drifted = p_var < alpha_var
                else:
                    f_stat = 1.0
                    p_var = 1.0
                    var_drifted = False

                if mean_drifted:
                    n_drifted_mean += 1
                if var_drifted:
                    n_drifted_var += 1

                variables[col] = {
                    "run_mean": round(run_mean, 6),
                    "run_std": round(run_std, 6),
                    "n_run": n_run,
                    "ess_run": round(float(ess_run), 2),
                    "rho1_run": round(float(rho1_run), 4),
                    "z_score": round(float(z_score), 4),
                    "p_mean": round(float(p_mean), 6),
                    "mean_drifted": bool(mean_drifted),
                    "f_stat": round(float(f_stat), 4),
                    "p_var": round(float(p_var), 6),
                    "var_drifted": bool(var_drifted),
                }

            per_run[str(rid)] = {
                "variables": variables,
                "n_mean_drifted": n_drifted_mean,
                "n_var_drifted": n_drifted_var,
                "n_variables": n_vars,
                "fraction_mean_drifted": round(
                    n_drifted_mean / max(n_vars, 1), 4
                ),
                "fraction_var_drifted": round(
                    n_drifted_var / max(n_vars, 1), 4
                ),
            }

        return {
            "per_run": per_run,
            "z_critical": round(float(z_crit), 4),
            "alpha_bonferroni_mean": round(float(alpha_mean), 8),
            "alpha_bonferroni_var": round(float(alpha_var), 8),
        }

    # ═══════════════════════════════════════════════════════════════════
    # Perspective 2: Correlation drift
    # ═══════════════════════════════════════════════════════════════════

    def _score_correlation(
        self,
        score_runs: Dict[Any, pd.DataFrame],
        score_ids: List[Any],
    ) -> Dict[str, Any]:
        """
        Compare each run's correlation matrix against the pooled
        Pearson correlation matrix.

        Two measures:
        1. Frobenius norm of (run_corr - pooled_corr), normalized
           by number of elements. Higher = more different.
        2. Per-element Fisher z-transformation test: for each pair
           (i, j), test whether the run's correlation differs from
           the pooled correlation.

        The Fisher-z test's sample sizes are autocorrelation-corrected
        (effective sample size, per pair — see
        _effective_sample_size_pair) rather than raw row counts, for
        the same reason as perspective 1: row-ordered data violates
        the i.i.d. assumption the test otherwise makes.
        """
        if self._pooled_corr is None:
            return {"status": "skipped", "reason": "no pooled correlations"}

        # Use columns present in the pooled correlation matrix
        corr_cols = [c for c in self._columns
                     if c in self._pooled_corr.columns]
        if len(corr_cols) < 2:
            return {"status": "skipped", "reason": "too few columns"}

        pooled_corr = self._pooled_corr.loc[corr_cols, corr_cols]
        n_pairs = len(corr_cols) * (len(corr_cols) - 1) // 2

        per_run = {}

        for rid in score_ids:
            run_df = score_runs[rid][corr_cols]
            run_corr = run_df.corr(method="pearson")
            n_run = len(run_df)

            # -- Frobenius norm (normalized) --
            diff = (run_corr.values - pooled_corr.values)
            # Zero out diagonal (always 1.0)
            np.fill_diagonal(diff, 0)
            frob_norm = float(np.linalg.norm(diff, "fro"))
            frob_normalized = frob_norm / max(n_pairs, 1)

            # -- Fisher z per-element test --
            n_pooled = self._baseline[corr_cols[0]]["pooled_n"]
            # Per-column lag-1 autocorrelation for this run, computed once
            # and reused for every pair below.
            run_rho1 = {
                c: self._lag1_autocorr(run_df[c].values.astype(np.float64))
                for c in corr_cols
            }
            n_drifted_pairs = 0
            top_drifted = []

            for i in range(len(corr_cols)):
                for j in range(i + 1, len(corr_cols)):
                    col_a, col_b = corr_cols[i], corr_cols[j]
                    r_run = run_corr.iloc[i, j]
                    r_pool = pooled_corr.iloc[i, j]

                    # Fisher z-transform: z = 0.5 * ln((1+r)/(1-r))
                    # Clamp to avoid log(0)
                    r_run_c = np.clip(r_run, -0.9999, 0.9999)
                    r_pool_c = np.clip(r_pool, -0.9999, 0.9999)

                    z_run = 0.5 * np.log((1 + r_run_c) / (1 - r_run_c))
                    z_pool = 0.5 * np.log((1 + r_pool_c) / (1 - r_pool_c))

                    # Autocorrelation-corrected effective sample sizes,
                    # in place of raw n_run/n_pooled (see module notes).
                    ess_run = self._effective_sample_size_pair(
                        n_run, run_rho1[col_a], run_rho1[col_b]
                    )
                    ess_pooled = self._effective_sample_size_pair(
                        n_pooled,
                        self._baseline[col_a]["mean_rho1"],
                        self._baseline[col_b]["mean_rho1"],
                    )

                    # SE of difference: sqrt(1/(n1-3) + 1/(n2-3))
                    if ess_run > 3 and ess_pooled > 3:
                        se = np.sqrt(1 / (ess_run - 3) + 1 / (ess_pooled - 3))
                        z_stat = (z_run - z_pool) / se
                        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                        drifted = p_val < self.corr_alpha
                    else:
                        z_stat = 0.0
                        p_val = 1.0
                        drifted = False

                    if drifted:
                        n_drifted_pairs += 1
                        top_drifted.append({
                            "var_a": corr_cols[i],
                            "var_b": corr_cols[j],
                            "r_run": round(float(r_run), 4),
                            "r_pooled": round(float(r_pool), 4),
                            "z_stat": round(float(z_stat), 4),
                            "p_value": round(float(p_val), 6),
                        })

            # Sort drifted pairs by absolute z_stat
            top_drifted.sort(key=lambda x: abs(x["z_stat"]), reverse=True)

            per_run[str(rid)] = {
                "frobenius_norm": round(frob_norm, 4),
                "frobenius_normalized": round(frob_normalized, 6),
                "n_drifted_pairs": n_drifted_pairs,
                "n_total_pairs": n_pairs,
                "fraction_drifted": round(
                    n_drifted_pairs / max(n_pairs, 1), 4
                ),
                "top_drifted_pairs": top_drifted[:20],
            }

        return {
            "status": "ok",
            "per_run": per_run,
            "n_columns": len(corr_cols),
            "n_pairs": n_pairs,
        }

    # ═══════════════════════════════════════════════════════════════════
    # Perspective 3: Causal drift
    # ═══════════════════════════════════════════════════════════════════

    def _score_causal(self) -> Dict[str, Any]:
        """
        Compare each run's causal links against the consensus set.

        Consensus links: links detected in >= consensus_threshold
        fraction of runs (default: all runs).

        For each run, report:
        - How many consensus links are missing in this run
        - How many extra links appear that are not consensus
        """
        if self._causal_aggregated is None:
            return {"status": "skipped", "reason": "no causal results"}

        link_freq = self._causal_aggregated.get("link_frequency", {})
        n_runs = self._causal_aggregated.get("n_runs", 1)
        per_run_results = self._causal_aggregated.get("per_run_results", [])

        if n_runs < 1 or not link_freq:
            return {"status": "skipped", "reason": "no causal links found"}

        # Determine consensus links
        min_count = int(np.ceil(self.consensus_threshold * n_runs))
        consensus_links = {
            k for k, count in link_freq.items()
            if count >= min_count
        }

        # Determine per-run link sets
        # Reconstruct from the aggregated data: a link is "present" in
        # a run if it was detected (significant) in that run.
        # We need to reconstruct this from per_run_results in the
        # causal_aggregated data. The per_run_results only has counts,
        # not the specific links. So we need the raw link data.
        #
        # Since we only have frequency counts, we can score each run
        # by what's available: links that appear in ALL runs vs those
        # that don't. For per-run membership, we check if the link
        # frequency < n_runs (meaning some run is missing it).
        #
        # More precise: we know how many runs each link appeared in.
        # A link appearing in (n_runs - 1) runs means exactly 1 run
        # is missing it. But we can't tell WHICH run without the raw
        # per-run data.
        #
        # Better approach: for each run, count links that appear in
        # fewer than n_runs runs as "unstable links" — they may or
        # may not be in this specific run.

        # Build a summary of consensus vs total links
        all_links = set(link_freq.keys())
        non_consensus = all_links - consensus_links

        mean_val = self._causal_aggregated.get("mean_val", {})

        # Per-run info from causal analysis
        per_run = {}
        for run_info in per_run_results:
            rid = run_info["run_id"]
            status = run_info.get("status", "unknown")

            if status != "ok":
                per_run[str(rid)] = {
                    "status": status,
                    "reason": run_info.get("reason", ""),
                }
                continue

            n_sig = run_info.get("n_significant", 0)

            per_run[str(rid)] = {
                "status": "ok",
                "n_significant_links": n_sig,
                "n_consensus_links": len(consensus_links),
                "n_total_unique_links": len(all_links),
            }

        # Top consensus links with their strength
        consensus_ranked = sorted(
            [(k, link_freq[k], abs(mean_val.get(k, 0)))
             for k in consensus_links],
            key=lambda x: x[2],
            reverse=True,
        )

        return {
            "status": "ok",
            "n_consensus_links": len(consensus_links),
            "n_total_unique_links": len(all_links),
            "n_non_consensus": len(non_consensus),
            "consensus_threshold": self.consensus_threshold,
            "min_run_count": min_count,
            "n_runs": n_runs,
            "consensus_links": [
                {
                    "link": k,
                    "frequency": freq,
                    "mean_parcorr": round(val, 4),
                }
                for k, freq, val in consensus_ranked[:50]
            ],
            "non_consensus_links": [
                {
                    "link": k,
                    "frequency": link_freq[k],
                    "mean_parcorr": round(abs(mean_val.get(k, 0)), 4),
                }
                for k in sorted(non_consensus,
                                key=lambda x: link_freq[x],
                                reverse=True)[:50]
            ],
            "per_run": per_run,
        }

    # ═══════════════════════════════════════════════════════════════════
    # Perspective 4: Onset detection (CUSUM)
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _cusum_arl0(h: float, k: float) -> float:
        """
        Siegmund's continuity-corrected approximation for the
        in-control (zero-shift) average run length of a one-sided
        CUSUM chart with reference value k and decision interval h.

        ARL0(h) = (exp(2*k*b) - 2*k*b - 1) / (2*k^2),  b = h + 1.166

        Ref: Siegmund, D. (1985), Sequential Analysis: Tests and
        Confidence Intervals; Montgomery, Introduction to Statistical
        Quality Control.
        """
        b = h + 1.166
        return (np.exp(2 * k * b) - 2 * k * b - 1) / (2 * k ** 2)

    @classmethod
    def _solve_cusum_h(cls, k: float, target_arl0: float) -> float:
        """
        Solve for the CUSUM decision interval h that yields the
        desired in-control average run length (target_arl0), given
        reference value k, via Siegmund's ARL approximation.
        """
        def f(h):
            return cls._cusum_arl0(h, k) - target_arl0

        lo, hi = 1e-6, 1.0
        while f(hi) < 0:
            hi *= 2
            if hi > 1e6:
                raise ValueError(
                    "Could not bracket a root for the CUSUM decision "
                    "interval — check cusum_shift_sigma/cusum_arl0."
                )
        return brentq(f, lo, hi)

    def _score_cusum(
        self,
        score_runs: Dict[Any, pd.DataFrame],
        score_ids: List[Any],
    ) -> Dict[str, Any]:
        """
        For each run and each variable, run a one-sided CUSUM control
        chart (Page, 1954) against the baseline mean/std to localize
        the first sample where a sustained shift begins.

        Standardized against the baseline: z_t = (x_t - mu0) / sigma0.

        Upper (increase) chart: C+_t = max(0, C+_{t-1} + z_t - k)
        Lower (decrease) chart: C-_t = min(0, C-_{t-1} + z_t + k)

        Onset = first sample where C+_t > h (increase) or
        C-_t < -h (decrease). k is fixed from cusum_shift_sigma; h is
        solved fresh here (not hand-picked) from an *effective* target
        ARL0 = cusum_arl0 * 2 * n_variables — the same Bonferroni-style
        idea used for stat_alpha/var_alpha elsewhere in this module,
        applied in ARL units: each variable runs two one-sided charts
        (increase/decrease), so without inflating the per-channel
        target, the chance of a false alarm somewhere across all
        variables compounds well past the intended cusum_arl0.

        Sorting variables by their onset sample within a run gives a
        first-pass propagation ordering (which variable moved first).
        """
        k = self.cusum_k
        cols = self._columns
        n_vars = len(cols)

        effective_arl0 = self.cusum_arl0 * 2 * max(n_vars, 1)
        h = self._solve_cusum_h(k, effective_arl0)

        mu0 = np.array([self._baseline[c]["pooled_mean"] for c in cols])
        sigma0 = np.array([self._baseline[c]["pooled_std"] for c in cols])
        valid = sigma0 > 1e-12

        per_run = {}

        for rid in score_ids:
            run_df = score_runs[rid][cols]
            data = run_df.values.astype(np.float64)
            sample_index = run_df.index.values
            n_samples = data.shape[0]

            # Standardize; treat missing/invalid-baseline columns as
            # neutral (z=0) so they neither trigger nor crash the chart.
            z = np.zeros_like(data)
            safe_sigma = np.where(valid, sigma0, 1.0)
            z = (data - mu0) / safe_sigma
            z[:, ~valid] = 0.0
            z[np.isnan(z)] = 0.0

            c_pos = np.zeros(n_vars)
            c_neg = np.zeros(n_vars)
            onset_pos = np.full(n_vars, -1, dtype=int)
            onset_dir = [None] * n_vars
            peak = np.zeros(n_vars)

            for t in range(n_samples):
                c_pos = np.maximum(0.0, c_pos + z[t] - k)
                c_neg = np.minimum(0.0, c_neg + z[t] + k)
                peak = np.maximum(peak, np.maximum(c_pos, -c_neg))

                newly_up = (onset_pos < 0) & (c_pos > h)
                newly_down = (onset_pos < 0) & (c_neg < -h)

                for i in np.where(newly_up)[0]:
                    onset_pos[i] = t
                    onset_dir[i] = "increase"
                for i in np.where(newly_down)[0]:
                    onset_pos[i] = t
                    onset_dir[i] = "decrease"

            variables = {}
            onsets = []
            for i, col in enumerate(cols):
                detected = valid[i] and onset_pos[i] >= 0
                onset_sample = (
                    int(sample_index[onset_pos[i]]) if detected else None
                )
                variables[col] = {
                    "onset_sample": onset_sample,
                    "onset_position": int(onset_pos[i]) if detected else None,
                    "onset_direction": onset_dir[i] if detected else None,
                    "peak_cusum": round(float(peak[i]), 4),
                    "drifted": bool(detected),
                }
                if detected:
                    onsets.append((col, onset_sample))

            onsets.sort(key=lambda x: x[1])

            per_run[str(rid)] = {
                "variables": variables,
                "n_variables": n_vars,
                "n_onset_detected": len(onsets),
                "fraction_onset_detected": round(
                    len(onsets) / max(n_vars, 1), 4
                ),
                "earliest_onset_sample": onsets[0][1] if onsets else None,
                "earliest_onset_variable": onsets[0][0] if onsets else None,
                "propagation_order": [
                    {"variable": c, "onset_sample": s} for c, s in onsets
                ],
            }

        return {
            "status": "ok",
            "per_run": per_run,
            "k": round(float(k), 4),
            "h": round(float(h), 4),
            "shift_sigma": self.cusum_shift_sigma,
            "target_arl0_family_wise": self.cusum_arl0,
            "effective_arl0_per_channel": round(float(effective_arl0), 2),
        }

    # ═══════════════════════════════════════════════════════════════════
    # Run summary
    # ═══════════════════════════════════════════════════════════════════

    def _build_run_summary(
        self,
        stat_results: Dict[str, Any],
        corr_results: Dict[str, Any],
        causal_results: Dict[str, Any],
        cusum_results: Dict[str, Any],
        score_ids: Optional[List[Any]] = None,
        quality_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a per-run anomaly summary combining all four perspectives.

        Anomaly score = weighted combination of:
        - fraction of variables with mean drift
        - fraction of variables with variance drift
        - fraction of drifted correlation pairs
        - fraction of variables with a detected CUSUM onset

        Every entry also carries "data_quality_gate"
        ("passed"|"warning"|"failed") and the full "data_quality"
        report. A run gated "failed" won't appear in stat_results/
        corr_results/cusum_results (score_runs() excludes it before
        computing them), so it falls through to this method's all-zero
        defaults below — anomaly_score stays 0.0 rather than a fake
        number, and "data_quality_gate": "failed" is the explicit,
        unmissable statement that no fault reasoning was actually done
        for this run, not a claim that it looked normal.
        """
        if score_ids is None:
            score_ids = self._run_ids
        quality_results = quality_results or {}

        summary = {}

        for rid in score_ids:
            rid_str = str(rid)
            entry = {"run_id": rid_str}

            quality = quality_results.get(rid_str)
            if quality:
                severity = quality["overall"]["severity"]
                entry["data_quality_gate"] = "failed" if severity == "critical" else (
                    "warning" if severity == "warning" else "passed"
                )
                entry["data_quality"] = quality["overall"]
            else:
                entry["data_quality_gate"] = "passed"
                entry["data_quality"] = None

            # Statistical drift
            if "per_run" in stat_results and rid_str in stat_results["per_run"]:
                sr = stat_results["per_run"][rid_str]
                entry["frac_mean_drifted"] = sr["fraction_mean_drifted"]
                entry["frac_var_drifted"] = sr["fraction_var_drifted"]
                entry["n_mean_drifted"] = sr["n_mean_drifted"]
                entry["n_var_drifted"] = sr["n_var_drifted"]

                # Top drifted variables by |z_score|
                drifted_vars = [
                    (col, v["z_score"])
                    for col, v in sr["variables"].items()
                    if v["mean_drifted"]
                ]
                drifted_vars.sort(key=lambda x: abs(x[1]), reverse=True)
                entry["top_mean_drifted"] = [
                    {"variable": col, "z_score": round(z, 4)}
                    for col, z in drifted_vars[:10]
                ]
            else:
                entry["frac_mean_drifted"] = 0.0
                entry["frac_var_drifted"] = 0.0

            # Correlation drift
            if (corr_results.get("status") == "ok"
                    and rid_str in corr_results.get("per_run", {})):
                cr = corr_results["per_run"][rid_str]
                entry["corr_frobenius"] = cr["frobenius_normalized"]
                entry["frac_corr_drifted"] = cr["fraction_drifted"]
            else:
                entry["corr_frobenius"] = 0.0
                entry["frac_corr_drifted"] = 0.0

            # CUSUM onset detection
            if (cusum_results.get("status") == "ok"
                    and rid_str in cusum_results.get("per_run", {})):
                cu = cusum_results["per_run"][rid_str]
                entry["frac_onset_detected"] = cu["fraction_onset_detected"]
                entry["earliest_onset_sample"] = cu["earliest_onset_sample"]
                entry["earliest_onset_variable"] = cu["earliest_onset_variable"]
                entry["propagation_order"] = cu["propagation_order"][:10]
            else:
                entry["frac_onset_detected"] = 0.0
                entry["earliest_onset_sample"] = None
                entry["earliest_onset_variable"] = None
                entry["propagation_order"] = []

            # Composite anomaly score (simple weighted sum)
            # Each component is 0-1, higher = more anomalous
            entry["anomaly_score"] = round(
                0.3 * entry["frac_mean_drifted"]
                + 0.15 * entry["frac_var_drifted"]
                + 0.25 * entry["frac_corr_drifted"]
                + 0.3 * entry["frac_onset_detected"],
                4,
            )

            summary[rid_str] = entry

        return summary