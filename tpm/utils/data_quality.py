"""
Data Quality
------------
Shared data-quality checks, used by both Layer 1 (profiling a raw run)
and Layer 3 (gating every incoming batch before drift/fault reasoning
runs on it at all). Lives in utils/ because it's infrastructure both
layers need, not analysis logic specific to either one.

IMPORTANT DISTINCTION this module exists to enforce: "data quality" is
not the same question as "is this an anomaly."
    - Data quality asks: can these numbers be trusted at all?
    - Anomaly/fault detection (Layer 3's main job) asks: assuming the
      numbers ARE trustworthy, is the process behaving abnormally?

This module deliberately stays minimal: one generic, dataset-agnostic
built-in check (missing values), plus an open-ended mechanism for
operator-supplied rules (utils/quality_rules.py) that capture domain
knowledge no generic statistic could infer on its own (e.g. "this
column is a fraction, so it should never leave 0-1"). An earlier
version of this module also tried to generically detect frozen
sensors, gross-outlier readings, index gaps, and unit/scale changes —
all removed: a genuine, large process fault can look statistically
identical to "lots of extreme points," so any fixed statistical cutoff
meant to catch glitches either fires on real faults (Layer 3's job, not
this module's) or has to be hand-tuned per dataset, which contradicts
this project's generalization goal. Rather than keep chasing calibration,
those checks are gone; a human-authored rule (via chat) is how
domain-specific expectations get enforced instead.

check_data_quality() ends in an explicit trust verdict ("trusted": bool,
"severity": "ok"|"warning"|"critical", "reasons": [...]) — never just an
unexplained pass/fail. Only "critical" gates downstream analysis
(Layer 3); "warning" is surfaced but doesn't block anything.

THRESHOLD REGISTRY:
    +------------------------+---------+----------+---------------------------+
    | Parameter              | Default | Grounded | Basis                     |
    +------------------------+---------+----------+---------------------------+
    | missing_critical_frac  | 0.5     | No       | Explicit product decision:|
    |                        |         |          | more than half a column's |
    |                        |         |          | values missing means any  |
    |                        |         |          | statistic computed on it  |
    |                        |         |          | is unreliable — flag for  |
    |                        |         |          | human review rather than  |
    |                        |         |          | silently proceeding.      |
    | missing_warning_frac   | 0.02    | No       | Heuristic "worth a note"  |
    |                        |         |          | floor — any missing data  |
    |                        |         |          | above incidental rounding.|
    +------------------------+---------+----------+---------------------------+
    Custom rules (utils/quality_rules.py) have no tunable threshold here:
    any violation of an operator-specified rule is reported, since the
    rule itself already encodes the operator's own tolerance.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from tpm.utils.quality_rules import evaluate_rules


def _check_missing(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    n = len(df)
    per_column = {}
    for col in columns:
        if col not in df.columns:
            per_column[col] = {"n_missing": n, "frac_missing": 1.0, "column_absent": True}
            continue
        n_missing = int(df[col].isna().sum())
        per_column[col] = {
            "n_missing": n_missing,
            "frac_missing": round(n_missing / n, 4) if n else 0.0,
            "column_absent": False,
        }
    worst = max((v["frac_missing"] for v in per_column.values()), default=0.0)
    return {"per_column": per_column, "max_frac_missing": round(worst, 4)}


def _build_verdict(
    missing: Dict[str, Any],
    rule_violations: List[Dict[str, Any]],
    missing_critical_frac: float,
    missing_warning_frac: float,
) -> Dict[str, Any]:
    reasons_critical = []
    reasons_warning = []

    if missing["max_frac_missing"] >= missing_critical_frac:
        worst_col = max(missing["per_column"], key=lambda c: missing["per_column"][c]["frac_missing"])
        reasons_critical.append(
            f"{worst_col} is {missing['per_column'][worst_col]['frac_missing']:.0%} missing "
            f"(>= {missing_critical_frac:.0%} threshold) — flagged for human review"
        )
    elif missing["max_frac_missing"] >= missing_warning_frac:
        reasons_warning.append(f"Up to {missing['max_frac_missing']:.1%} missing values in at least one column")

    for v in rule_violations:
        reasons_critical.append(
            f"{v['column']} violated rule \"{v['description']}\": {v['n_flagged']} sample(s) "
            f"out of range — flagged for human review"
        )

    if reasons_critical:
        severity, trusted = "critical", False
    elif reasons_warning:
        severity, trusted = "warning", True
    else:
        severity, trusted = "ok", True

    return {
        "trusted": trusted,
        "severity": severity,
        "reasons": reasons_critical + reasons_warning,
    }


def check_data_quality(
    df: pd.DataFrame,
    columns: List[str],
    custom_rules: Optional[List[Dict[str, Any]]] = None,
    *,
    missing_critical_frac: float = 0.5,
    missing_warning_frac: float = 0.02,
) -> Dict[str, Any]:
    """
    Run data-quality checks on one run/batch's DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        One run's (or one incoming batch's) data.
    columns : list of str
        Columns to check.
    custom_rules : list of dict, optional
        Operator-authored rules (utils/quality_rules.py's schema) to
        evaluate against `df` — typically `quality_rules.load_active_rules(...)`
        for this dataset.

    Returns
    -------
    dict
        {"n_samples":, "missing":, "rule_violations":, "overall": {
        "trusted": bool, "severity": "ok"|"warning"|"critical",
        "reasons": [str, ...]}}
    """
    missing = _check_missing(df, columns)
    rule_violations = evaluate_rules(df, custom_rules or [])
    overall = _build_verdict(missing, rule_violations, missing_critical_frac, missing_warning_frac)

    return {
        "n_samples": len(df),
        "missing": missing,
        "rule_violations": rule_violations,
        "overall": overall,
    }
