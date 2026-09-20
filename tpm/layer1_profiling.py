"""
Layer 1: Statistical Profiling
------------------------------
Thin wrapper around ydata-profiling. All per-column stats, correlations,
and alerts come directly from the library — no custom calculations added.

Profiles each run separately to get per-run statistical fingerprints.
For a *pooled* profile across multiple runs (recommended for anything
shown as "what does normal look like" — e.g. a dashboard — rather than
one arbitrary run's fingerprint), concatenate the runs first and call
profile() once on the combined DataFrame; nothing here needs to know
about "runs" as a concept, only whatever DataFrame it's given.

Also runs the shared data-quality checks (utils/data_quality.py) on the
raw run itself — missing values, plus any operator-authored custom
rules for this dataset (utils/quality_rules.py) — since a fingerprint of
untrustworthy data is misleading regardless of how it's computed. See
that module's docstring for why this is a distinct question from
anomaly detection (Layer 3's job).

Interface:
    Input:  a single DataFrame (one run) from Layer 0
    Output: ProfileReport (ydata-profiling object)

    profiler = StatisticalProfiler()
    report = profiler.profile(df, columns)
    quality = profiler.check_quality(df, columns)

    # Export to JSON (native ydata-profiling format)
    report.to_file("profile.json")

    # Programmatic access for downstream layers
    desc = report.get_description()
    desc.variables          → dict {col_name: {46+ fields}}
    desc.correlations       → dict {method: pd.DataFrame}
    desc.alerts              → list of alert strings
    desc.table              → dict {n, n_var, memory_size, ...}
"""

import warnings
import pandas as pd
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

from tpm.utils.data_quality import check_data_quality

try:
    from ydata_profiling import ProfileReport
except ImportError:
    raise ImportError(
        "ydata-profiling is required for Layer 1. "
        "Install with: pip install ydata-profiling"
    )


class StatisticalProfiler:
    """
    Layer 1 profiler — thin wrapper around ydata-profiling.

    All output comes directly from the library. No custom fields added.
    Profiles one run at a time.

    Usage:
        profiler = StatisticalProfiler()
        report = profiler.profile(df, columns)
        quality = profiler.check_quality(df, columns)
        if not quality["overall"]["trusted"]:
            print("Data quality issue:", quality["overall"]["reasons"])

        # Native JSON export
        report.to_file("profile.json")

        # Programmatic access (interface for downstream layers)
        desc = report.get_description()
        desc.variables["col_a"]["mean"]      → float
        desc.correlations["pearson"]        → pd.DataFrame
        desc.alerts                         → list
    """

    def __init__(
        self,
        correlations: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        correlations : list of str, optional
            Which correlation types to compute.
            Default: ["pearson", "spearman", "kendall", "phi_k", "auto"]
            Available: pearson, spearman, kendall, phi_k, cramers, auto
        """
        self.correlation_types = correlations or [
            "pearson", "spearman", "kendall", "phi_k", "auto"
        ]
        self._last_report: Optional[ProfileReport] = None

    def profile(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        tsmode: bool = True,
        title: str = "Layer 1 Profile",
    ) -> ProfileReport:
        """
        Profile a single run's columns.

        Parameters
        ----------
        df : pd.DataFrame
            One run's data from Layer 0.
        columns : list of str, optional
            Columns to profile. If None, uses all columns in df.
        tsmode : bool
            Enable time-series analysis (stationarity, seasonality via ADF).
        title : str
            Report title (stored in output metadata).

        Returns
        -------
        ProfileReport
            The ydata-profiling report object.
            - report.to_file("out.json") → native JSON export
            - report.get_description()   → programmatic access
        """
        if columns is not None:
            df = df[columns]

        # Build correlation config
        all_corr_types = ["pearson", "spearman", "kendall", "phi_k", "cramers", "auto"]
        corr_config = {
            ctype: {"calculate": ctype in self.correlation_types}
            for ctype in all_corr_types
        }

        report = ProfileReport(
            df,
            tsmode=tsmode,
            title=title,
            correlations=corr_config,
            progress_bar=False,
            minimal=False,
            interactions={"continuous": False},  # skip scatter plots
            samples={"head": 0, "tail": 0},     # skip sample rows
            duplicates={"head": 0},              # skip duplicate listing
        )

        # Force computation so get_description() is populated
        report.get_description()

        self._last_report = report
        return report

    def check_quality(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        custom_rules: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run the shared data-quality checks (utils/data_quality.py) on
        this run's raw data: a missing-values check, plus any
        operator-authored custom rules for this dataset — ending in an
        explicit trust verdict.

        This is a distinct question from anomaly detection (Layer 3):
        it asks whether the numbers can be trusted at all, not whether
        the process they describe is behaving abnormally. See
        utils/data_quality.py's module docstring for why that
        distinction matters and isn't just semantic.

        Parameters
        ----------
        df : pd.DataFrame
            One run's raw data (same DataFrame you'd pass to profile()).
        columns : list of str, optional
            Columns to check. Defaults to all columns in df.
        custom_rules : list of dict, optional
            Operator-authored rules for this dataset (see
            utils/quality_rules.py), typically
            `quality_rules.load_active_rules(dataset_dir)`.
        **kwargs
            Forwarded to check_data_quality() (missing_critical_frac,
            missing_warning_frac).

        Returns
        -------
        dict
            See utils/data_quality.check_data_quality()'s return shape.
        """
        cols = columns if columns is not None else list(df.columns)
        return check_data_quality(df, cols, custom_rules=custom_rules, **kwargs)

    @property
    def report(self) -> Optional[ProfileReport]:
        """Access the last profiling report."""
        return self._last_report