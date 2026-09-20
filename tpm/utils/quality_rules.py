"""
Quality Rules
-------------
User-authored data-quality rules — the operator-knowledge counterpart to
utils/data_quality.py's one built-in check (missing values). A rule
captures something a human knows about a column that no generic
statistical test could infer from the data alone (e.g. "this column is
a fraction, so it should never leave 0-1").

Rules are authored in natural language through the dashboard chat
(see layer4_llm.py's LLMAnalyst.chat, which gives the model a
`propose_data_quality_rule` tool it can call on its own judgment) and
always require a human to confirm the LLM's translation before
`save_rule` is ever called — translating a rule from English happens
once, under human review; applying it to every future batch afterward
is the pure, deterministic `evaluate_rules` below, with no LLM
involved at check time.

Only one rule type exists today: "range" (an optional min and/or max
bound on one column) — the type field exists so more kinds can be
added later without changing the storage format.

Storage: one JSON file per dataset, `quality_rules.json` in that
dataset's `analysis_output/<name>/` directory (same convention as
every other per-dataset artifact in this project).
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

RULE_TYPES = {"range"}


def _rules_path(dataset_dir: str) -> str:
    return os.path.join(dataset_dir, "quality_rules.json")


def load_rules(dataset_dir: str) -> List[Dict[str, Any]]:
    """All rules for this dataset, active or not. Empty list if none saved yet."""
    path = _rules_path(dataset_dir)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("rules", [])


def load_active_rules(dataset_dir: str) -> List[Dict[str, Any]]:
    """Rules that should actually be applied by check_data_quality()."""
    return [r for r in load_rules(dataset_dir) if r.get("active", True)]


def save_rule(dataset_dir: str, rule: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persist a human-confirmed rule. Assigns rule_id/created_at/active if
    not already set, appends to this dataset's rule set, and writes it
    back out. Returns the saved rule (with its assigned fields).
    """
    rules = load_rules(dataset_dir)
    saved = dict(rule)
    saved.setdefault("rule_id", uuid.uuid4().hex[:12])
    saved.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    saved.setdefault("active", True)
    rules.append(saved)
    _write_rules(dataset_dir, rules)
    return saved


def delete_rule(dataset_dir: str, rule_id: str) -> None:
    rules = [r for r in load_rules(dataset_dir) if r.get("rule_id") != rule_id]
    _write_rules(dataset_dir, rules)


def _write_rules(dataset_dir: str, rules: List[Dict[str, Any]]) -> None:
    os.makedirs(dataset_dir, exist_ok=True)
    with open(_rules_path(dataset_dir), "w", encoding="utf-8") as f:
        json.dump({"rules": rules}, f, indent=2)


def validate_rule_shape(rule: Dict[str, Any], columns: List[str]) -> Optional[str]:
    """
    Sanity-check a proposed rule before it's ever shown to a human for
    confirmation (or saved). Returns None if valid, else a short
    human-readable reason it isn't — e.g. because the LLM named a
    column that doesn't actually exist in this dataset, or gave neither
    bound, both of which would otherwise silently produce a useless or
    broken rule.
    """
    column = rule.get("column")
    if not column:
        return "no column was specified"
    if column not in columns:
        return f"'{column}' is not a known column in this dataset"

    rule_type = rule.get("type", "range")
    if rule_type not in RULE_TYPES:
        return f"unsupported rule type '{rule_type}'"

    if rule_type == "range":
        lo, hi = rule.get("min"), rule.get("max")
        if lo is None and hi is None:
            return "no lower or upper bound was given"
        if lo is not None and hi is not None and lo > hi:
            return f"min ({lo}) is greater than max ({hi})"

    if not rule.get("description") and not rule.get("plain_english"):
        return "no plain-English description was given"

    return None


def evaluate_rules(df: pd.DataFrame, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply each active rule to `df` and report any violations. Pure,
    deterministic, no LLM involved — the LLM's only job was authoring
    the rule once, under human review (see module docstring).

    Returns
    -------
    list of {"rule_id", "column", "description", "n_flagged", "frac_flagged"}
        One entry per rule that was actually violated (rules with zero
        violations are omitted).
    """
    violations = []
    for rule in rules:
        if not rule.get("active", True):
            continue
        column = rule.get("column")
        if column not in df.columns:
            continue

        series = df[column].dropna()
        if len(series) == 0:
            continue

        rule_type = rule.get("type", "range")
        if rule_type == "range":
            lo, hi = rule.get("min"), rule.get("max")
            mask = pd.Series(False, index=series.index)
            if lo is not None:
                mask |= series < lo
            if hi is not None:
                mask |= series > hi
        else:
            continue

        n_flagged = int(mask.sum())
        if n_flagged > 0:
            violations.append({
                "rule_id": rule.get("rule_id"),
                "column": column,
                "description": rule.get("description") or rule.get("plain_english") or "",
                "n_flagged": n_flagged,
                "frac_flagged": round(n_flagged / len(series), 4),
            })

    return violations
