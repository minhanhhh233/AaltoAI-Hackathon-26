"""
Pipeline Service
------------------
Orchestrates Layers 0-4 for the two web operations. Pure orchestration —
no new analysis logic lives here, only calls into the existing layer/
utils modules.

    build_baseline()  Layer 0 -> 1 -> 2 -> 4 (column identities, variable
                       types, cluster roles) on a random sample of a
                       dataset's runs, exported via ResultsExporter and
                       plotted via PipelinePlotter. Runs inline in
                       whatever thread calls it — tpm/web/jobs.py wraps
                       this in a background thread so the HTTP request
                       returns immediately.

    check_run()        Reloads a built baseline from disk (the same
                       PipelinePlotter.load_from_folder() pattern the
                       plotting module already uses to avoid keeping
                       pipeline state in memory), scores one uploaded
                       run with Layer 3, runs Layer 4's
                       root_cause_analysis, and renders that run's plots.
                       Fast enough (no PCMCI, no per-run profiling) to
                       run synchronously within a request.

Never processes more than a bounded, explicit number of runs in
build_baseline — stated in the result, not hidden — since this app runs
Layer 1/2 inline rather than in a distributed job queue.
"""

import json
import os
import random
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from tpm.layer0_ingestion import DataIngestion
from tpm.layer1_profiling import StatisticalProfiler
from tpm.layer2_relational import CausalAnalyzer
from tpm.layer3_drift import DriftDetector
from tpm.layer4_llm import LLMAnalyst, column_stats_from_layer1_summary, correlations_from_df
from tpm.utils.results_exporter import ResultsExporter
from tpm.utils.plotting import PipelinePlotter
from tpm.utils.llm_client import get_default_client, get_reasoning_client
from tpm.utils import quality_rules

from .datasets import get_dataset

BASELINE_DONE_MARKER = "_build_complete.json"
ANALYSIS_OUTPUT_DIR = "analysis_output"

ProgressFn = Callable[..., None]


def _analysis_dir(dataset_name: str) -> str:
    return os.path.join(ANALYSIS_OUTPUT_DIR, dataset_name)


def is_baseline_ready(dataset_name: str) -> bool:
    return os.path.exists(os.path.join(_analysis_dir(dataset_name), BASELINE_DONE_MARKER))


def get_build_info(dataset_name: str) -> Optional[Dict[str, Any]]:
    """Read back the marker written at the end of build_baseline(), if any."""
    path = os.path.join(_analysis_dir(dataset_name), BASELINE_DONE_MARKER)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _noop_progress(step: str, current: int = 0, total: int = 0, message: str = ""):
    pass


def _read_json(output_dir: str, *parts) -> Optional[Any]:
    path = os.path.join(output_dir, *parts)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


WEB_CAUSAL_TAU_MAX = 3
"""
Layer 2's own default (tau_max=10, see layer2_relational.py) is tuned for
thorough, offline causal discovery, not an interactive web request.
Measured live: PCMCI at tau_max=10 across this dataset's 52 variables did
not finish even one run in 5+ minutes. tau_max=3 is the tradeoff for this
web service specifically — Layer 2 itself is unchanged for batch/offline
use where that thoroughness is worth the wait.
"""


def build_baseline(
    dataset_name: str,
    n_runs: int = 3,
    include_causal: bool = True,
    seed: Optional[int] = None,
    progress: Optional[ProgressFn] = None,
    causal_tau_max: int = WEB_CAUSAL_TAU_MAX,
) -> Dict[str, Any]:
    """
    Build (or rebuild) a dataset's baseline analysis: Layers 0-2 on a
    random sample of `n_runs` runs, Layer 4 column/cluster
    interpretation, then export + plot everything.
    """
    report = progress or _noop_progress
    dataset = get_dataset(dataset_name)

    report("loading", message=f"Loading {os.path.basename(dataset['filepath'])}")
    ingestion = DataIngestion()
    data = ingestion.load_multi_run(dataset["filepath"])
    all_run_ids = data["run_ids"]

    rng = random.Random(seed)
    n_runs = min(n_runs, len(all_run_ids))
    chosen_ids = sorted(rng.sample(all_run_ids, n_runs))
    runs = {rid: data["runs"][rid] for rid in chosen_ids}
    columns = data["columns"]

    exporter = ResultsExporter(dataset_name)
    exporter.export_dataset_meta(data["metadata"], data["labels"])
    exporter.export_run_data(runs, columns)
    audit_logger = exporter.get_audit_logger()

    # ── Layer 1: statistical profiling + data quality ───────────────
    report("profiling", 0, len(runs), "Profiling runs")
    profiler = StatisticalProfiler()
    custom_rules = quality_rules.load_active_rules(exporter.output_dir)
    run_descriptions = {}
    quality_by_run = {}
    for i, rid in enumerate(chosen_ids):
        rep = profiler.profile(runs[rid], columns)
        exporter.export_layer1(rid, rep)
        run_descriptions[rid] = rep.get_description()
        quality = profiler.check_quality(runs[rid], columns, custom_rules=custom_rules)
        quality_by_run[rid] = quality
        severity = quality["overall"]["severity"]
        if severity in ("critical", "warning"):
            # Logged for both severities (not just critical) since the
            # Flags tab is now the only place this surfaces at all — the
            # dashboard no longer shows an inline summary banner.
            audit_logger.log_decision(
                component="layer1.data_quality",
                decision="flagged for human review" if severity == "critical" else "noted — minor data quality issue",
                evidence={"run_id": str(rid), "severity": severity, "reasons": quality["overall"]["reasons"]},
            )
        report("profiling", i + 1, len(runs), f"Profiled run {rid}")

    exporter.export_layer1_summary(run_descriptions)
    exporter.export_layer1_quality(quality_by_run)

    # Pooled profile (concatenate the sampled runs, profile once) — the
    # dashboard's "what does normal look like overall" view, consistent
    # with Layer 3's own baseline being pooled-across-runs rather than
    # any single run (see export_layer1_pooled_profile's docstring).
    report("profiling", message="Building pooled profile across sampled runs")
    pooled_df = pd.concat([runs[rid][columns] for rid in chosen_ids], ignore_index=True)
    # tsmode=False: the pooled profile is only used for the dashboard's
    # per-column stats/histogram (export_layer1_pooled_profile extracts
    # a fixed set of plain descriptive fields — none of them ADF/
    # stationarity-derived). Time-series diagnostics on a large pooled
    # series (many runs concatenated) is expensive and memory-heavy for
    # no benefit here; per-run profiles above still run with tsmode on.
    pooled_report = profiler.profile(pooled_df, columns, tsmode=False, title="Pooled Profile")
    exporter.export_layer1_pooled_profile(pooled_report.get_description())

    # ── Layer 2: correlation, clustering, causal discovery ──────────
    report("correlating", message="Computing pooled correlations")
    # tau_max only affects analyze_per_run() below; compute_pooled_correlations
    # and cluster_columns() ignore it, so one analyzer instance covers both.
    analyzer = CausalAnalyzer(tau_max=causal_tau_max)
    pooled_corr = analyzer.compute_pooled_correlations(runs, columns)
    exporter.export_layer2_correlations(pooled_corr)

    report("clustering", message="Clustering variables")
    clusters = analyzer.cluster_columns(pooled_corr, method="spearman")
    exporter.export_layer2_clusters(clusters)

    causal = None
    if include_causal:
        report("causal_discovery", 0, len(runs), "Running PCMCI per run (slowest step)")
        causal = analyzer.analyze_per_run(runs, columns)
        exporter.export_layer2_causal(causal)
        report("causal_discovery", len(runs), len(runs), "Causal discovery complete")

    # ── Layer 4: LLM interpretation ─────────────────────────────────
    report("llm_interpretation", message="Inferring variable identities")
    analyst = LLMAnalyst(get_default_client(), get_reasoning_client(), audit_logger)

    layer1_summary_shape = {
        "runs": {str(rid): {"variables": dict(run_descriptions[rid].variables)} for rid in chosen_ids}
    }
    column_stats = column_stats_from_layer1_summary(layer1_summary_shape)
    corr_dict = correlations_from_df(pooled_corr["pearson"])

    identities = analyst.infer_column_identities(columns, column_stats, corr_dict, clusters, causal)
    report("llm_interpretation", message="Classifying measured vs. manipulated variables")
    var_types = analyst.infer_variable_type(columns, column_stats, causal, identities)
    report("llm_interpretation", message="Proposing cluster role hypotheses")
    cluster_roles = analyst.propose_cluster_roles(clusters, identities, column_stats)

    exporter.export_layer4_llm({
        "column_identities": identities,
        "variable_types": var_types,
        "cluster_roles": cluster_roles,
    })

    # ── Report + figures ─────────────────────────────────────────────
    report("reporting", message="Writing report")
    exporter.write_report(
        metadata=data["metadata"],
        run_descriptions=run_descriptions,
        pooled_correlations=pooled_corr,
        clusters=clusters,
        causal_aggregated=causal,
    )

    report("plotting", message="Rendering figures")
    plotter = PipelinePlotter(os.path.join(exporter.output_dir, "figures"))
    plotter.generate_all(
        runs=runs,
        columns=columns,
        pooled_correlations=pooled_corr,
        clusters=clusters,
        causal_aggregated=causal,
    )

    build_info = {
        "n_runs": len(runs),
        "run_ids": [str(r) for r in chosen_ids],
        "total_runs_in_dataset": len(all_run_ids),
        "include_causal": include_causal,
    }
    with open(os.path.join(exporter.output_dir, BASELINE_DONE_MARKER), "w", encoding="utf-8") as f:
        json.dump(build_info, f, indent=2)

    report("done", len(runs), len(runs), "Baseline analysis complete")
    return build_info


def load_dashboard_context(dataset_name: str) -> Dict[str, Any]:
    """
    Read back everything build_baseline() wrote, in a template-friendly
    shape (figure URLs already resolved, LLM cards pre-flattened) for
    tpm/web/templates/dataset.html.
    """
    output_dir = _analysis_dir(dataset_name)
    build_info = get_build_info(dataset_name) or {}

    dataset_meta = _read_json(output_dir, "dataset_meta.json") or {}
    clusters = _read_json(output_dir, "layer2", "clusters.json") or {}
    llm = _read_json(output_dir, "layer4", "llm_inference.json") or {}
    pooled_profile = _read_json(output_dir, "layer1", "pooled_profile.json") or {}

    figures_url = f"/figures/{dataset_name}/figures"

    def fig(name: str) -> Optional[str]:
        path = os.path.join(output_dir, "figures", name)
        return f"{figures_url}/{name}" if os.path.exists(path) else None

    def safe_name(col: str) -> str:
        return col.replace("/", "_").replace(" ", "_").replace(".", "_")

    def var_fig(subfolder: str, col: str) -> Optional[str]:
        filename = f"{safe_name(col)}.png"
        path = os.path.join(output_dir, "figures", subfolder, filename)
        return f"{figures_url}/{subfolder}/{filename}" if os.path.exists(path) else None

    identities = (llm.get("column_identities") or {}).get("columns", {})
    var_types = (llm.get("variable_types") or {}).get("columns", {})
    cluster_roles = (llm.get("cluster_roles") or {}).get("clusters", {})
    pooled_columns = pooled_profile.get("columns", {})

    def format_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
        stats = dict(stats)
        mem = stats.get("memory_size")
        if isinstance(mem, (int, float)):
            stats["memory_size_display"] = f"{mem / 1024:.1f} KiB" if mem >= 1024 else f"{mem:.0f} B"
        return stats

    rules = quality_rules.load_active_rules(output_dir)
    rules_by_column: Dict[str, list] = {}
    for r in rules:
        rules_by_column.setdefault(r["column"], []).append(r)

    column_cards = []
    for col in dataset_meta.get("columns", []):
        column_cards.append({
            "column": col,
            "identity": identities.get(col, {}),
            "variable_type": var_types.get(col, {}),
            "stats": format_stats(pooled_columns.get(col, {})),
            "histogram_url": var_fig("var_histograms", col),
            "overlay_url": var_fig("var_overlays", col),
            "rules": rules_by_column.get(col, []),
        })

    cluster_cards = []
    for cid, members in clusters.get("clusters", {}).items():
        cluster_cards.append({
            "cluster_id": cid,
            "members": members,
            "role": cluster_roles.get(str(cid), {}),
        })

    report_path = os.path.join(output_dir, "report.md")
    report_text = ""
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            report_text = f.read()

    return {
        "build_info": build_info,
        "dataset_meta": dataset_meta,
        "column_cards": column_cards,
        "cluster_cards": cluster_cards,
        "report_text": report_text,
        "figures": {
            "clustermap": fig("clustermap_pearson.png"),
            "heatmap": fig("corr_heatmap_pearson.png"),
            "causal_graph": fig("causal_graph.png"),
            "causal_frequency": fig("causal_frequency.png"),
            "causal_lag_profiles": fig("causal_lag_profiles.png"),
        },
    }


def check_run(dataset_name: str, uploaded_filepath: str, run_label: str = "uploaded") -> Dict[str, Any]:
    """
    Score one uploaded run against a dataset's already-built baseline.
    Raises RuntimeError if the baseline hasn't been built yet.
    """
    if not is_baseline_ready(dataset_name):
        raise RuntimeError(
            f"'{dataset_name}' has no baseline analysis yet — build it first."
        )

    output_dir = _analysis_dir(dataset_name)
    baseline_data = PipelinePlotter.load_from_folder(output_dir)
    baseline_runs = baseline_data["runs"]
    columns = baseline_data["columns"]
    pooled_corr = baseline_data["pooled_correlations"]
    causal = baseline_data["causal_aggregated"]

    ingestion = DataIngestion()
    new_df = ingestion.load(uploaded_filepath)
    missing = [c for c in columns if c not in new_df.columns]
    if missing:
        raise ValueError(
            f"Uploaded file is missing {len(missing)} expected variable(s), "
            f"e.g. {missing[:8]}. Expected the same variables as the baseline dataset."
        )
    new_df = new_df[columns]

    custom_rules = quality_rules.load_active_rules(output_dir)
    detector = DriftDetector()
    detector.build_baseline(baseline_runs, columns, pooled_correlations=pooled_corr, causal_aggregated=causal)
    drift_results = detector.score_runs(test_runs={run_label: new_df}, custom_rules=custom_rules)
    run_summary = drift_results["run_summary"]
    gate_failed = run_summary[run_label]["data_quality_gate"] == "failed"
    severity = _severity_for_score(run_summary[run_label].get("anomaly_score", 0))
    # Root cause ("which components might be wrong") only makes sense to
    # ask an LLM to explain when the run actually looks anomalous — a
    # "good" (normal) or "warning" (mildly off, not clearly a fault)
    # severity has nothing that needs explaining, so skip the LLM call
    # and the section entirely rather than manufacturing a hypothesis
    # for a run that isn't actually showing a fault.
    needs_root_cause = severity in ("serious", "critical")

    if gate_failed:
        ResultsExporter(dataset_name).get_audit_logger().log_decision(
            component="layer3.data_quality_gate",
            decision="flagged for human review",
            evidence={"run_label": run_label, "reasons": run_summary[run_label]["data_quality"]["reasons"]},
        )

    figures_url = f"/figures/{dataset_name}/figures"
    if gate_failed:
        # Data quality gate failed: no drift/fault reasoning was done for
        # this run (see layer3_drift.py), so there's nothing meaningful
        # to plot or ask an LLM to explain — skip both rather than
        # rendering an empty chart or a confusing "no anomaly found"
        # hypothesis for a run that was never actually analyzed.
        root_cause = None
        figures = {"breakdown": None, "propagation": None, "statistical": None, "variables": None}
    else:
        if needs_root_cause:
            llm = _read_llm_inference(output_dir)
            identities = llm.get("column_identities") if llm else None
            analyst = LLMAnalyst(get_default_client(), get_reasoning_client())
            root_cause = analyst.root_cause_analysis(run_label, run_summary[run_label], causal, identities)
        else:
            root_cause = None

        plotter = PipelinePlotter(os.path.join(output_dir, "figures"))
        fig_files = {
            "breakdown": plotter.plot_run_drift_breakdown(run_summary, [run_label]),
            "propagation": plotter.plot_cusum_propagation(run_summary, run_label),
            "statistical": plotter.plot_statistical_drift_bar(run_summary, run_label),
            "variables": plotter.plot_variable_with_onset(
                {run_label: new_df}, run_label, drift_results["baseline"], run_summary,
            ),
        }
        figures = {
            key: (f"{figures_url}/{os.path.relpath(path, os.path.join(output_dir, 'figures'))}"
                  .replace(os.sep, "/") if path else None)
            for key, path in fig_files.items()
        }

    return {
        "run_label": run_label,
        "run_summary": run_summary[run_label],
        "severity": severity,
        "drift_results": drift_results,
        "root_cause": root_cause,
        "figures": figures,
    }


def _severity_for_score(score: float) -> str:
    """Same visualization bands as utils/plotting.py's plot_anomaly_ranking
    (good/warning/serious/critical) — a chart-legend convention, not a
    statistical threshold; kept consistent between the plot and this page."""
    if score < 0.15:
        return "good"
    if score < 0.35:
        return "warning"
    if score < 0.6:
        return "serious"
    return "critical"


def _read_llm_inference(output_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(output_dir, "layer4", "llm_inference.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_report_text(dataset_name: str) -> str:
    path = os.path.join(_analysis_dir(dataset_name), "report.md")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def get_dataset_columns(dataset_name: str) -> List[str]:
    path = os.path.join(_analysis_dir(dataset_name), "dataset_meta.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("columns", [])


def confirm_quality_rule(dataset_name: str, rule: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persist a rule a human has just confirmed (typically proposed by
    LLMAnalyst.chat()'s tool call — see layer4_llm.py). Re-validates
    against this dataset's real columns before saving, since the
    request could in principle be tampered with between proposal and
    confirmation.
    """
    columns = get_dataset_columns(dataset_name)
    error = quality_rules.validate_rule_shape(rule, columns)
    if error:
        raise ValueError(f"Can't save this rule: {error}")
    return quality_rules.save_rule(_analysis_dir(dataset_name), rule)


def delete_quality_rule(dataset_name: str, rule_id: str) -> None:
    quality_rules.delete_rule(_analysis_dir(dataset_name), rule_id)


def get_flags(dataset_name: str) -> List[Dict[str, Any]]:
    """
    Data-quality flags raised for this dataset so far — every
    `log_decision(...)` call Layer 1/3/4 made when a run or chat
    message was flagged for human review, newest first.
    """
    records = ResultsExporter(dataset_name).get_audit_logger().read_all()
    flags = [r for r in records if r.get("type") == "decision"]
    return list(reversed(flags))


def _column_top_correlations(output_dir: str, column: str, top_n: int = 6) -> Dict[str, float]:
    corr_path = os.path.join(output_dir, "layer2", "correlations_pearson.csv")
    if not os.path.exists(corr_path):
        return {}
    corr_df = pd.read_csv(corr_path, index_col=0)
    if column not in corr_df.columns:
        return {}
    s = corr_df[column].drop(labels=[column], errors="ignore").dropna()
    top = s.reindex(s.abs().sort_values(ascending=False).index)[:top_n]
    return {str(k): round(float(v), 4) for k, v in top.items()}


def explain_figure(dataset_name: str, figure_type: str, target: Optional[str] = None) -> Dict[str, Any]:
    """
    "Explain this" button support: builds the same derived-data bundle a
    given dashboard chart was rendered from, and asks Layer 4 to explain
    it in plain language (LLMAnalyst.explain_figure — never given the
    rendered image itself, only these numbers).
    """
    output_dir = _analysis_dir(dataset_name)
    llm = _read_json(output_dir, "layer4", "llm_inference.json") or {}
    pooled_profile = _read_json(output_dir, "layer1", "pooled_profile.json") or {}
    clusters = _read_json(output_dir, "layer2", "clusters.json") or {}
    causal = _read_json(output_dir, "layer2", "causal_per_run.json") or {}

    if figure_type == "variable_profile":
        if not target:
            raise ValueError("A variable name is required to explain its profile.")
        data = {
            "variable": target,
            "stats": pooled_profile.get("columns", {}).get(target, {}),
            "cluster_id": clusters.get("column_cluster", {}).get(target),
            "top_correlations": _column_top_correlations(output_dir, target),
        }
        label = f"Distribution histogram and time-series trace for variable {target!r}"

    elif figure_type == "correlation_heatmap":
        corr_path = os.path.join(output_dir, "layer2", "correlations_pearson.csv")
        corr_df = pd.read_csv(corr_path, index_col=0) if os.path.exists(corr_path) else None
        data = correlations_from_df(corr_df) if corr_df is not None else {}
        label = "Heatmap of the strongest pairwise Pearson correlations among variables"

    elif figure_type == "clustermap":
        data = {
            "best_k": clusters.get("best_k"),
            "best_silhouette": clusters.get("best_silhouette"),
            "clusters": clusters.get("clusters", {}),
        }
        label = "Hierarchical clustering dendrogram of variables, colored by cluster assignment"

    elif figure_type == "causal_graph":
        data = {
            "n_runs": causal.get("n_runs"),
            "link_frequency": causal.get("link_frequency", {}),
            "mean_val": causal.get("mean_val", {}),
        }
        label = "Causal graph (PCMCI) of consensus links detected across analyzed runs"

    else:
        raise ValueError(f"Unknown figure type: {figure_type!r}")

    analyst = LLMAnalyst(get_default_client())
    return analyst.explain_figure(label, data)
