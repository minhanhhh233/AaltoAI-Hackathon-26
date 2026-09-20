"""
Layer 4: LLM-Assisted Interpretation
-------------------------------------
Uses an LLM to generate *hypotheses* about a dataset's semantics and
about the root cause of detected anomalies, grounded in the derived
statistical artifacts produced by Layers 1-3 — never in raw data.

Four analytical tasks, plus a chat interface for interactive follow-up:

    1. infer_column_identities()  — what each unlabeled column likely
       represents, using general domain knowledge (typical ranges/
       behavior of common physical or business quantities) combined
       with this dataset's own statistics/correlations/causal role.
    2. infer_variable_type()      — measured/observed vs. manipulated/
       controlled, if that distinction exists in the data at all.
    3. propose_cluster_roles()    — a plausible common functional role
       for each correlated column cluster (from Layer 2's
       cluster_columns()) — explicitly a hypothesis-generation aid,
       not a deterministic result like Layers 1-3's statistics.
    4. root_cause_analysis()      — ranked root-cause hypotheses for a
       drifted run (from Layer 3), each with an explicit justification
       (onset timing, causal-graph consistency, drift magnitude) —
       not just an unexplained aggregate score.
    5. chat()                     — stateless multi-turn Q&A grounded in
       the same derived artifacts (e.g. utils/results_exporter's
       report.md). Also offers a `propose_data_quality_rule` tool the
       model can call on its own judgment when a user describes a
       checkable rule for a column — always surfaced as a proposal for
       human confirmation (utils/quality_rules.py), never saved
       automatically.
    6. explain_figure()           — on-demand plain-language explanation
       of one specific dashboard chart, grounded in the same derived
       data the chart was rendered from (never the image itself).

CRITICAL CONSTRAINT: the LLM never sees raw data rows — only derived
statistical fingerprints, correlation/cluster summaries, and causal
graphs (all plain, JSON-serializable dicts, never a DataFrame). This is
enforced in code, not just documented: every public method rejects any
argument (recursively, through dicts/lists) that turns out to be a
pandas DataFrame/Series.

Generalization: no dataset- or domain-specific assumptions are baked
into the prompts (no hardcoded sensor names, units, or "this is process
X" logic) — the LLM is asked to infer or state uncertainty about the
domain itself, the same way for any dataset.

Epistemic status, important distinction from Layers 1-3: everything
this layer produces is an LLM-generated hypothesis with a
self-reported, subjective "confidence" — NOT a calibrated statistical
confidence interval like Layers 1-3's p-values/z-scores. Every method's
output carries an explicit disclaimer field saying so; a consuming
report/UI should visually separate this layer's output from Layers
1-3's deterministic results, never blend them into one number.

Auditability: every external model call — exactly what was sent, to
which model, and why — is recorded via an optional AuditLogger (see
utils/audit_log.py), not just the final structured result. Layers 0-3's
decisions are already fully auditable (every threshold/z-score/p-value
they use is deterministic and already saved in full); this is the
equivalent for Layer 4's non-deterministic, externally-billed calls.

Model routing: root_cause_analysis (the one task that must weigh
conflicting evidence rather than extract/classify) can use a separate,
stronger `reasoning_client` — see utils/llm_client.py's
get_reasoning_client(). Defaults to the same client as everything else
if not given.

Interface contract:
    from tpm.utils.llm_client import get_default_client, get_reasoning_client
    from tpm.utils.results_exporter import ResultsExporter

    exporter = ResultsExporter("my_dataset")
    analyst = LLMAnalyst(
        get_default_client(),
        reasoning_client=get_reasoning_client(),   # optional, stronger model
        audit_logger=exporter.get_audit_logger(),  # optional, full call trail
    )

    # Bridge helpers turn Layer 1/2 output into Layer 4's plain-dict
    # inputs (these are the only functions here that touch a DataFrame
    # — solely to strip it away before anything reaches the LLM):
    column_stats = column_stats_from_layer1_summary(layer1_summary)
    correlations = correlations_from_df(pooled_corr_df)

    identities = analyst.infer_column_identities(
        columns, column_stats, correlations, clusters, causal_aggregated,
    )
    var_types = analyst.infer_variable_type(
        columns, column_stats, causal_aggregated, identities,
    )
    cluster_roles = analyst.propose_cluster_roles(clusters, identities)
    root_causes = analyst.root_cause_analysis(
        run_id, run_summary[run_id], causal_aggregated, identities,
    )
    reply, history, pending_rule = analyst.chat(
        "Why is run 42 anomalous?", history=[], context=report_md_text, columns=columns,
    )
"""

import functools
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tpm.utils.quality_rules import validate_rule_shape


# ═══════════════════════════════════════════════════════════════════════
# "Never raw data" guard — enforced, not just documented
# ═══════════════════════════════════════════════════════════════════════

def _reject_dataframes(value: Any, path: str = "argument"):
    """Recursively raise if `value` contains a pandas DataFrame/Series."""
    if isinstance(value, (pd.DataFrame, pd.Series)):
        raise TypeError(
            f"{path} is a pandas {type(value).__name__} — Layer 4 must never "
            f"receive raw data rows, only derived summaries/statistics. Pass "
            f"a dict/JSON-serializable summary instead (see the "
            f"column_stats_from_layer1_summary / correlations_from_df "
            f"bridge helpers in this module)."
        )
    if isinstance(value, dict):
        for k, v in value.items():
            _reject_dataframes(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _reject_dataframes(v, f"{path}[{i}]")


def _no_raw_data(fn):
    """Decorator: reject any DataFrame/Series among a method's arguments."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        for i, a in enumerate(args):
            _reject_dataframes(a, f"positional arg {i}")
        for k, v in kwargs.items():
            _reject_dataframes(v, k)
        return fn(self, *args, **kwargs)
    return wrapper


# ═══════════════════════════════════════════════════════════════════════
# Bridge helpers — the only functions here that touch a DataFrame,
# specifically to strip it away into a plain dict before anything
# reaches an LLMAnalyst method.
# ═══════════════════════════════════════════════════════════════════════

def column_stats_from_layer1_summary(layer1_summary: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Average per-column statistics across runs from utils/
    results_exporter's layer1/summary.json shape
    ({"runs": {run_id: {"variables": {col: {...}}}}}) into a single
    pooled column_stats dict, suitable for infer_column_identities /
    infer_variable_type.
    """
    runs = layer1_summary.get("runs", {})
    acc: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for run_data in runs.values():
        for col, stats in run_data.get("variables", {}).items():
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    acc[col][k].append(v)
    return {
        col: {k: float(np.mean(vals)) for k, vals in stat_lists.items() if vals}
        for col, stat_lists in acc.items()
    }


def correlations_from_df(corr_df: pd.DataFrame, top_n: int = 8) -> Dict[str, Dict[str, float]]:
    """
    Convert a pooled correlation matrix (Layer 2's compute_pooled_
    correlations) into a plain nested dict of {col: {other_col: r}},
    keeping only the top_n strongest partners per column — keeps
    Layer 4's interface pandas-free and JSON-serializable.
    """
    cols = corr_df.columns.tolist()
    result = {}
    for c in cols:
        s = corr_df[c].drop(labels=[c], errors="ignore").dropna()
        if s.empty:
            continue
        top = s.reindex(s.abs().sort_values(ascending=False).index)[:top_n]
        result[c] = {str(k): round(float(v), 4) for k, v in top.items()}
    return result


# ═══════════════════════════════════════════════════════════════════════
# Shared prompt scaffolding
# ═══════════════════════════════════════════════════════════════════════

GROUND_RULES = """You are assisting with interpreting a statistical/causal analysis of a
dataset. You are NOT given raw data rows — only derived statistics,
correlations, cluster assignments, and causal-graph summaries computed
by deterministic methods. Follow these rules strictly:

1. Treat every number given to you as ground truth. Never invent, alter,
   or "correct" a number; never fabricate a statistic, correlation, or
   relationship not present in the input.
2. You may use general domain knowledge (e.g. typical ranges or behavior
   of common physical or business quantities) to form hypotheses, but
   every hypothesis must cite the specific evidence (from the input)
   that supports it.
3. Clearly distinguish hypothesis from fact. Never state a hypothesis as
   if it were a measured/verified result.
4. The dataset's domain/industry is not told to you. Infer it cautiously
   from the evidence, or say it's unclear — do not assume a specific
   familiar dataset if the evidence doesn't clearly indicate one.
5. If the evidence is insufficient to support any confident hypothesis
   for something, say so plainly rather than guessing.
"""


def _strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _parse_json_response(text: str) -> Dict[str, Any]:
    """
    Parse an LLM's JSON response defensively: strips a markdown code
    fence if present (models sometimes add one despite instructions
    not to), and returns a clearly-flagged error payload — preserving
    the raw text for debugging — rather than raising, so one malformed
    response doesn't take down a batch job.
    """
    cleaned = _strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return {"error": "could_not_parse_llm_response", "raw_response": text}


_DISCLAIMER = (
    "LLM-generated hypothesis using general domain knowledge and the "
    "evidence given below — not a deterministic statistical result, and "
    "not independently verified. Confidence is the model's own subjective "
    "self-assessment, not a calibrated statistical measure."
)


# A tool chat() offers on every turn, letting the model decide on its own
# judgment when a user's message describes a checkable rule for a column
# (see utils/quality_rules.py) — it never saves anything itself, only
# drafts a proposal chat() then surfaces for a human to confirm or reject.
QUALITY_RULE_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_data_quality_rule",
        "description": (
            "Call this when the user describes an expected value range or "
            "threshold for a specific data column (e.g. 'xmeas_7 should "
            "always be between 0 and 100'). This only drafts a rule for a "
            "human to confirm — it is never saved automatically, so call "
            "it whenever such a statement comes up rather than just "
            "answering in prose."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "The exact column name this rule applies to.",
                },
                "min": {
                    "type": ["number", "null"],
                    "description": "Lower bound, or null if the user only gave an upper bound.",
                },
                "max": {
                    "type": ["number", "null"],
                    "description": "Upper bound, or null if the user only gave a lower bound.",
                },
                "plain_english": {
                    "type": "string",
                    "description": "Restate the rule in plain English, for a human to confirm.",
                },
            },
            "required": ["column", "plain_english"],
        },
    },
}


class LLMAnalyst:
    """
    Layer 4 — LLM-assisted interpretation of Layers 1-3's derived
    statistical artifacts. Provider-agnostic: takes any LLMClient (see
    utils/llm_client.py), so the backend (OpenAI API, a local model,
    another provider) is swappable without touching this class.

    Two clients, one purpose split: `client` handles the simpler,
    classification-shaped tasks (column identity, variable type,
    cluster roles); `reasoning_client` (defaults to `client` if not
    given) handles root_cause_analysis, which must weigh conflicting
    evidence rather than extract/classify — a good place to spend a
    stronger (pricier) model if you have one configured. See
    utils/llm_client.py's get_default_client() / get_reasoning_client().

    If `audit_logger` is given (see utils/audit_log.py — typically
    obtained from ResultsExporter.get_audit_logger()), every external
    model call is recorded: exactly what was sent, to which model, why,
    and what came back — the trail behind every hypothesis this layer
    produces.
    """

    def __init__(self, client, reasoning_client=None, audit_logger=None):
        self.client = client
        self.reasoning_client = reasoning_client or client
        self.audit_logger = audit_logger

    # ═══════════════════════════════════════════════════════════════════
    # Shared evidence-building helpers
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _causal_degree(
        causal: Optional[Dict[str, Any]],
        columns: List[str],
        min_frequency_frac: float = 0.5,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Per-column causal in/out-degree from Layer 2's aggregated
        per-run link frequencies, keeping only links detected in at
        least min_frequency_frac of runs (majority-consensus edges) so
        single-run noise isn't treated as structural evidence.
        """
        if not causal:
            return {}
        link_freq = causal.get("link_frequency", {})
        n_runs = causal.get("n_runs", 1) or 1

        out_deg, in_deg = defaultdict(int), defaultdict(int)
        out_partners, in_partners = defaultdict(set), defaultdict(set)
        for key, count in link_freq.items():
            if count / n_runs < min_frequency_frac:
                continue
            src, tgt, _tau = key.split("|")
            if src == tgt:
                continue
            out_deg[src] += 1
            in_deg[tgt] += 1
            out_partners[src].add(tgt)
            in_partners[tgt].add(src)

        return {
            col: {
                "out_degree": out_deg.get(col, 0),
                "in_degree": in_deg.get(col, 0),
                "drives": sorted(out_partners.get(col, [])),
                "driven_by": sorted(in_partners.get(col, [])),
            }
            for col in columns
        }

    def _build_column_evidence(
        self,
        columns: List[str],
        column_stats: Dict[str, Dict[str, Any]],
        correlations: Optional[Dict[str, Dict[str, float]]] = None,
        clusters: Optional[Dict[str, Any]] = None,
        causal: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        degree = self._causal_degree(causal, columns) if causal else {}
        column_cluster = (clusters or {}).get("column_cluster", {})

        evidence = {}
        for col in columns:
            entry: Dict[str, Any] = {"stats": column_stats.get(col, {})}
            if correlations and col in correlations:
                top = sorted(correlations[col].items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
                entry["top_correlations"] = [{"column": c, "r": r} for c, r in top]
            if col in column_cluster:
                entry["cluster_id"] = column_cluster[col]
            if col in degree:
                entry["causal"] = degree[col]
            evidence[col] = entry
        return evidence

    def _call_json(
        self,
        system: str,
        user_content: str,
        *,
        task: str,
        purpose: str,
        client=None,
    ) -> Dict[str, Any]:
        client = client or self.client
        raw = client.complete(
            system=system,
            messages=[{"role": "user", "content": user_content}],
            json_mode=True,
        )
        parsed = _parse_json_response(raw)
        if self.audit_logger:
            self.audit_logger.log_llm_call(
                task=task,
                model=getattr(client, "model", "unknown"),
                purpose=purpose,
                system_prompt=system,
                user_content=user_content,
                response=raw,
                parsed_result=parsed,
            )
        return parsed

    # ═══════════════════════════════════════════════════════════════════
    # 1. Column identity inference
    # ═══════════════════════════════════════════════════════════════════

    @_no_raw_data
    def infer_column_identities(
        self,
        columns: List[str],
        column_stats: Dict[str, Dict[str, Any]],
        correlations: Optional[Dict[str, Dict[str, float]]] = None,
        clusters: Optional[Dict[str, Any]] = None,
        causal: Optional[Dict[str, Any]] = None,
        known_labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Hypothesize what each column likely represents, grounded in its
        own statistics, correlated partners, cluster membership, and
        causal role — combined with general domain knowledge.

        Returns
        -------
        dict
            {"disclaimer": str, "columns": {col: {"hypothesis": str,
            "confidence": "low"|"medium"|"high", "evidence_cited":
            [str, ...], "reasoning": str}}}
        """
        evidence = self._build_column_evidence(columns, column_stats, correlations, clusters, causal)

        system = GROUND_RULES + """
TASK: For each column in the evidence below, hypothesize what it most
likely measures or represents. Use general domain knowledge (typical
ranges, units, or behavior patterns of common physical/business
quantities) together with the column's own statistics, its most
correlated partners, its cluster membership, and its causal role
(what it drives / is driven by, if any).

Respond with a JSON object of exactly this shape:
{"<column>": {"hypothesis": "<short phrase>", "confidence": "low"|"medium"|"high",
"evidence_cited": ["<specific number or relationship from the input>", ...],
"reasoning": "<1-3 sentences>"}, ...}
One entry per column listed in the evidence. Respond with JSON only.
"""
        user_content = "Column evidence (derived statistics only — no raw data rows):\n\n"
        user_content += json.dumps(evidence, indent=2, default=str)
        if known_labels:
            user_content += "\n\nAlready-known column labels (for context, not to be repeated):\n"
            user_content += json.dumps(known_labels, indent=2)

        result = self._call_json(
            system, user_content,
            task="infer_column_identities",
            purpose=f"Hypothesize identity for {len(columns)} column(s): {columns}",
        )
        return {"disclaimer": _DISCLAIMER, "columns": result}

    # ═══════════════════════════════════════════════════════════════════
    # 2. Measured vs. manipulated
    # ═══════════════════════════════════════════════════════════════════

    @_no_raw_data
    def infer_variable_type(
        self,
        columns: List[str],
        column_stats: Dict[str, Dict[str, Any]],
        causal: Optional[Dict[str, Any]] = None,
        column_identities: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Hypothesize whether each column is more likely a measured/
        observed quantity (an output of the system, typically driven by
        other variables) or a manipulated/controlled one (an input
        deliberately set, typically driving other variables) — if that
        distinction exists in this dataset at all.

        Returns
        -------
        dict
            {"disclaimer": str, "columns": {col: {"type":
            "measured"|"manipulated"|"uncertain"|"not_applicable",
            "confidence":, "reasoning":, "evidence_cited": [...]}}}
        """
        degree = self._causal_degree(causal, columns) if causal else {}
        identities = (column_identities or {}).get("columns", {})

        evidence = {}
        for col in columns:
            entry: Dict[str, Any] = {"stats": column_stats.get(col, {})}
            if col in degree:
                entry["causal"] = degree[col]
            if col in identities:
                entry["hypothesized_identity"] = identities[col].get("hypothesis")
            evidence[col] = entry

        system = GROUND_RULES + """
TASK: For each column, hypothesize whether it is more likely a MEASURED
quantity (an observed output of the system, typically driven by other
variables — high in-degree, low out-degree in the causal evidence) or a
MANIPULATED quantity (an input deliberately set/adjusted, typically
driving other variables — high out-degree, low in-degree). Some
datasets have no manipulated variables at all (e.g. pure observational
data) — if the evidence doesn't support this distinction for a column,
or for the dataset as a whole, say "uncertain" or "not_applicable"
rather than forcing a guess.

Respond with a JSON object of exactly this shape:
{"<column>": {"type": "measured"|"manipulated"|"uncertain"|"not_applicable",
"confidence": "low"|"medium"|"high",
"evidence_cited": ["<specific number or relationship>", ...],
"reasoning": "<1-3 sentences>"}, ...}
One entry per column. Respond with JSON only.
"""
        user_content = "Column evidence (derived statistics/causal structure only — no raw data rows):\n\n"
        user_content += json.dumps(evidence, indent=2, default=str)

        result = self._call_json(
            system, user_content,
            task="infer_variable_type",
            purpose=f"Classify measured vs. manipulated for {len(columns)} column(s): {columns}",
        )
        return {"disclaimer": _DISCLAIMER, "columns": result}

    # ═══════════════════════════════════════════════════════════════════
    # 3. Cluster-level functional role hypotheses
    # ═══════════════════════════════════════════════════════════════════

    @_no_raw_data
    def propose_cluster_roles(
        self,
        clusters: Dict[str, Any],
        column_identities: Optional[Dict[str, Any]] = None,
        column_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Propose a plausible common functional role/subsystem for each
        cluster of correlated columns (Layer 2's cluster_columns()
        output). Explicitly a hypothesis-generation aid — clusters are
        a deterministic statistical result, but what they collectively
        *mean* is not.

        Returns
        -------
        dict
            {"disclaimer": str, "clusters": {cluster_id: {"role_hypothesis":,
            "confidence":, "reasoning":}}}
        """
        if not clusters or clusters.get("status") != "ok":
            return {"disclaimer": _DISCLAIMER, "clusters": {}, "note": "no clustering result provided"}

        identities = (column_identities or {}).get("columns", {})
        evidence = {}
        for cid, members in clusters.get("clusters", {}).items():
            member_evidence = []
            for col in members:
                m: Dict[str, Any] = {"column": col}
                if col in identities:
                    m["hypothesized_identity"] = identities[col].get("hypothesis")
                if column_stats and col in column_stats:
                    m["stats"] = column_stats[col]
                member_evidence.append(m)
            evidence[str(cid)] = {"members": member_evidence}

        system = GROUND_RULES + """
TASK: Each cluster below is a group of columns that are strongly
correlated with each other (determined by deterministic hierarchical
clustering on their correlation structure — that part is already
established fact). For each cluster, propose a plausible common
functional role, subsystem, or theme these columns might collectively
represent. This is a hypothesis-generation aid only: state it as such,
and prefer "the evidence doesn't clearly suggest a common theme" over
forcing a role onto columns that don't obviously share one.

Respond with a JSON object of exactly this shape:
{"<cluster_id>": {"role_hypothesis": "<short phrase>",
"confidence": "low"|"medium"|"high", "reasoning": "<1-3 sentences>"}, ...}
One entry per cluster. Respond with JSON only.
"""
        user_content = "Cluster evidence:\n\n" + json.dumps(evidence, indent=2, default=str)
        result = self._call_json(
            system, user_content,
            task="propose_cluster_roles",
            purpose=f"Propose functional-role hypotheses for {len(evidence)} cluster(s)",
        )
        return {"disclaimer": _DISCLAIMER, "clusters": result}

    # ═══════════════════════════════════════════════════════════════════
    # 4. Root cause analysis
    # ═══════════════════════════════════════════════════════════════════

    @_no_raw_data
    def root_cause_analysis(
        self,
        run_id: Any,
        run_summary_entry: Dict[str, Any],
        causal_aggregated: Optional[Dict[str, Any]] = None,
        column_identities: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Rank root-cause hypotheses for one anomalous run (Layer 3's
        run_summary[run_id]), each with an explicit justification —
        not just the aggregate anomaly_score.

        Uses: the CUSUM onset/propagation order (temporal precedence —
        which variable moved first), the causal graph among the
        implicated variables (does the causal structure corroborate or
        challenge treating the earliest mover as the cause?), and drift
        magnitude (z-scores) as evidence to weigh against each other —
        the LLM is explicitly told these can conflict and to say so
        rather than mechanically picking the earliest onset.

        Returns
        -------
        dict
            {"disclaimer": str, "run_id":, "hypotheses": [{"rank":,
            "hypothesis":, "implicated_variables": [...], "confidence":,
            "supporting_evidence": [...], "caveats":}, ...]}
        """
        propagation_order = run_summary_entry.get("propagation_order", [])
        implicated = [p["variable"] for p in propagation_order]
        identities = (column_identities or {}).get("columns", {})

        causal_edges = []
        if causal_aggregated and implicated:
            link_freq = causal_aggregated.get("link_frequency", {})
            mean_val = causal_aggregated.get("mean_val", {})
            n_runs = causal_aggregated.get("n_runs", 1) or 1
            implicated_set = set(implicated)
            for key, count in link_freq.items():
                src, tgt, tau = key.split("|")
                if src in implicated_set and tgt in implicated_set and src != tgt:
                    causal_edges.append({
                        "source": src, "target": tgt, "lag": int(tau),
                        "detected_in_runs": f"{count}/{n_runs}",
                        "mean_partial_corr": round(mean_val.get(key, 0), 4),
                    })

        evidence = {
            "run_id": str(run_id),
            "anomaly_score": run_summary_entry.get("anomaly_score"),
            "drift_by_perspective": {
                "mean_shift_fraction": run_summary_entry.get("frac_mean_drifted"),
                "variance_fraction": run_summary_entry.get("frac_var_drifted"),
                "correlation_fraction": run_summary_entry.get("frac_corr_drifted"),
                "onset_fraction": run_summary_entry.get("frac_onset_detected"),
            },
            "onset_propagation_order_earliest_first": propagation_order,
            "top_mean_drifted_variables": run_summary_entry.get("top_mean_drifted", []),
            "causal_edges_among_implicated_variables": causal_edges,
            "hypothesized_identities": {
                v: identities[v].get("hypothesis") for v in implicated if v in identities
            },
        }

        system = GROUND_RULES + """
TASK: This run was flagged as anomalous. Propose ranked root-cause
hypotheses for WHERE the anomaly most likely originated and HOW it
propagated, using the evidence below:
- onset_propagation_order_earliest_first: which variable's statistics
  first departed from baseline, in order (from a CUSUM changepoint
  detector) — the earliest mover is a natural root-cause candidate, but
  is not automatically the cause: a downstream sensor can sometimes
  react faster than an upstream one, or a shared external cause can
  affect multiple variables near-simultaneously.
- causal_edges_among_implicated_variables: causal links (from
  independent per-run causal discovery on historical baseline data)
  among the variables that drifted in THIS run. If the earliest-onset
  variable causally drives the later-onset ones, that corroborates it
  as the root cause; if the causal graph suggests otherwise (e.g. the
  later variable actually drives the earlier one, or there's no causal
  link between them at all), say so explicitly — do not paper over the
  conflict.
- drift magnitude (z-scores, fractions) as supporting/weakening evidence.

Rank hypotheses by strength of support. If evidence conflicts or is
weak, reflect that in "confidence" and say so in "caveats" rather than
presenting a single confident answer.

Respond with a JSON object of exactly this shape:
{"hypotheses": [{"rank": 1, "hypothesis": "<1-2 sentences>",
"implicated_variables": ["<col>", ...], "confidence": "low"|"medium"|"high",
"supporting_evidence": ["<specific fact from the input>", ...],
"caveats": "<conflicting or weak evidence, or empty string>"}, ...]}
Respond with JSON only.
"""
        user_content = "Run evidence:\n\n" + json.dumps(evidence, indent=2, default=str)
        result = self._call_json(
            system, user_content,
            task="root_cause_analysis",
            purpose=f"Root cause analysis for run {run_id}",
            client=self.reasoning_client,
        )
        return {"disclaimer": _DISCLAIMER, "run_id": str(run_id), **result}

    # ═══════════════════════════════════════════════════════════════════
    # 5. Interactive chat (for the future web interface)
    # ═══════════════════════════════════════════════════════════════════

    @_no_raw_data
    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        context: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> Tuple[str, List[Dict[str, str]], Optional[Dict[str, Any]]]:
        """
        One turn of grounded Q&A about the analysis. Stateless by
        design (the caller supplies and persists `history`) so a web
        backend can serve many concurrent chat sessions from one
        LLMAnalyst instance without cross-talk.

        `context` is typically the text of utils/results_exporter's
        report.md (or an equivalent bundle of derived-artifact JSON) —
        the same "facts only, hypotheses clearly labeled" document a
        person would read, so the chat can't answer from anything the
        static report couldn't already justify.

        Every turn also offers the model QUALITY_RULE_TOOL — if the
        model decides (on its own judgment, not by a keyword match here)
        that the user just described a checkable rule for a column, it
        calls the tool instead of just replying in prose. That call is
        NEVER saved automatically: it's validated against the dataset's
        real columns (utils/quality_rules.validate_rule_shape) and
        returned as `pending_rule` for a human to confirm or reject —
        confirming is a separate, explicit action (see
        utils/quality_rules.save_rule). If the backend's LLMClient
        doesn't implement tool-calling at all, this degrades gracefully
        to a plain conversational reply with no rule proposal.

        Parameters
        ----------
        message : str
            The new user turn.
        history : list of {"role": "user"|"assistant", "content": str}
            Prior turns, oldest first. Pass [] (or None) for a new chat.
        context : str, optional
            Grounding material (derived artifacts only — never raw data).
        columns : list of str, optional
            This dataset's known columns — lets the model reference real
            column names and lets a proposed rule be validated against
            them before it's ever shown to a human.

        Returns
        -------
        (reply, new_history, pending_rule) : (str, list, dict or None)
            new_history is history + this exchange, ready to pass into
            the next call. pending_rule is a proposed rule dict awaiting
            human confirmation, or None on an ordinary chat turn.
        """
        history = list(history or [])

        system = GROUND_RULES + """
You are answering questions about a process-monitoring analysis in an
interactive chat. Ground every answer in the CONTEXT below. If the
context doesn't contain enough information to answer, say so rather
than guessing or using outside knowledge about specific numbers.

If the user describes an expected value range or threshold for a
specific column, call the propose_data_quality_rule tool instead of
just describing it in prose — that's how such rules get saved for
future data-quality checks. Only call it for a genuine, checkable rule
about one known column; for anything else, just reply normally.
"""
        if context:
            system += "\n\nCONTEXT (derived analysis artifacts — no raw data rows):\n\n" + context
        if columns:
            system += f"\n\nKnown columns in this dataset: {columns}"

        messages = history + [{"role": "user", "content": message}]

        try:
            result = self.client.complete_with_tools(
                system=system, messages=messages, tools=[QUALITY_RULE_TOOL],
            )
        except (NotImplementedError, AttributeError):
            # Either the backend doesn't support tool-calling (default
            # LLMClient.complete_with_tools behavior), or it's a client
            # that doesn't define the method at all (e.g. a minimal
            # duck-typed test double) — both degrade the same way: a
            # plain conversational reply with no rule proposal.
            result = {"role": "text", "content": self.client.complete(system=system, messages=messages, json_mode=False)}

        pending_rule = None
        if result["role"] == "tool_call" and result["tool_name"] == "propose_data_quality_rule":
            args = result["arguments"]
            candidate = {
                "column": args.get("column"),
                "type": "range",
                "min": args.get("min"),
                "max": args.get("max"),
                "description": args.get("plain_english"),
                "source_text": message,
            }
            error = validate_rule_shape(candidate, columns or [])
            if error is None:
                pending_rule = candidate
                reply = result.get("content") or (
                    f"I've drafted a rule from that: \"{candidate['description']}\". "
                    f"Confirm below to save it, or just tell me if that's not right."
                )
            else:
                reply = f"I tried to turn that into a rule, but {error}. Could you clarify?"
        else:
            reply = result["content"]

        if self.audit_logger:
            self.audit_logger.log_llm_call(
                task="chat",
                model=getattr(self.client, "model", "unknown"),
                purpose=f"Interactive chat turn: {message[:80]!r}",
                system_prompt=system,
                user_content=message,
                response=reply,
            )
            if pending_rule:
                self.audit_logger.log_decision(
                    component="layer4.chat_rule_proposal",
                    decision="proposed data-quality rule from chat, pending human confirmation",
                    evidence=pending_rule,
                    reasoning=result.get("content"),
                )

        new_history = messages + [{"role": "assistant", "content": reply}]
        return reply, new_history, pending_rule

    # ═══════════════════════════════════════════════════════════════════
    # 6. Explain a specific chart (dashboard "Explain this" buttons)
    # ═══════════════════════════════════════════════════════════════════

    @_no_raw_data
    def explain_figure(self, figure_label: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain what one specific chart on the dashboard shows, grounded
        in the same derived data used to render it. This method never
        sees the rendered image itself — like every other method here,
        it only ever receives derived statistics/summaries, never raw
        data or pixels — so the explanation comes from the same numbers
        a person could read off the chart's own axes/legend, not a
        literal description of what's drawn.

        Parameters
        ----------
        figure_label : str
            A short description of which chart this is (e.g. "Hierarchical
            clustering dendrogram of variables, colored by cluster").
        data : dict
            The derived data the chart was built from (stats, correlation
            values, cluster membership, causal link frequencies, etc.).

        Returns
        -------
        dict
            {"disclaimer": str, "explanation": str}
        """
        system = GROUND_RULES + f"""
TASK: A user clicked "Explain this" on a chart described as: {figure_label!r}.
Using ONLY the data below (the same data the chart was rendered from),
explain in plain language what this specific chart shows and what's
notable about it — 3-5 concise sentences, referring to actual values or
names present in the data rather than a generic description of what
this chart type usually looks like. If the data is sparse or empty, say
so rather than inventing detail.

Respond with a JSON object of exactly this shape:
{{"explanation": "<3-5 sentences>"}}
Respond with JSON only.
"""
        user_content = "Chart data:\n\n" + json.dumps(data, indent=2, default=str)
        result = self._call_json(
            system, user_content,
            task="explain_figure",
            purpose=f"Explain chart: {figure_label}",
        )
        return {"disclaimer": _DISCLAIMER, **result}
