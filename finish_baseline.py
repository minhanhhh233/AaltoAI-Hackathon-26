"""
Finish a baseline that was built with process_runs.py (Layers 0-2 only)
into something the web dashboard can show: pooled profile, Layer 4 LLM
interpretation, report.md, figures, and the "ready" marker.

Reuses the already-computed Layer 1/2 output on disk (correlations,
clusters, causal discovery, per-run data) rather than recomputing any
of it — this is purely the finishing steps process_runs.py skipped.

Usage (from the project root):

    python finish_baseline.py --dataset-name te_process_train_100 --output-dir analysis_output
"""

import argparse
import json
import os

import pandas as pd

from tpm.layer1_profiling import StatisticalProfiler
from tpm.layer4_llm import LLMAnalyst, column_stats_from_layer1_summary, correlations_from_df
from tpm.utils.llm_client import get_default_client, get_reasoning_client
from tpm.utils.plotting import PipelinePlotter
from tpm.utils.results_exporter import ResultsExporter
from tpm.web.pipeline_service import BASELINE_DONE_MARKER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True, help="Subfolder name under --output-dir (matches process_runs.py's --dataset-name)")
    parser.add_argument("--output-dir", default="analysis_output", help="Base output dir (default: ./analysis_output)")
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.dataset_name)
    if not os.path.isdir(output_dir):
        raise SystemExit(f"{output_dir} doesn't exist — run process_runs.py first.")

    print(f"[load] reloading existing Layer 0-2 output from {output_dir}")
    data = PipelinePlotter.load_from_folder(output_dir)
    runs = data["runs"]
    columns = data["columns"]
    pooled_corr = data["pooled_correlations"]
    clusters = data["clusters"]
    causal = data["causal_aggregated"]
    run_ids = sorted(runs.keys())

    with open(os.path.join(output_dir, "dataset_meta.json"), encoding="utf-8") as f:
        dataset_meta = json.load(f)
    with open(os.path.join(output_dir, "layer1", "summary.json"), encoding="utf-8") as f:
        layer1_summary_shape = json.load(f)

    exporter = ResultsExporter(args.dataset_name, base_dir=args.output_dir)
    audit_logger = exporter.get_audit_logger()

    # ── Pooled profile (Layer 1's "what does normal look like overall") ──
    print("[layer1] building pooled profile across all runs")
    profiler = StatisticalProfiler()
    pooled_df = pd.concat([runs[rid][columns] for rid in run_ids], ignore_index=True)
    # tsmode=False: avoids an expensive/memory-heavy ADF stationarity
    # test on the large pooled series — export_layer1_pooled_profile
    # only reads plain descriptive fields, none of them ADF-derived.
    pooled_report = profiler.profile(pooled_df, columns, tsmode=False, title="Pooled Profile")
    exporter.export_layer1_pooled_profile(pooled_report.get_description())

    # ── Layer 4: LLM interpretation ──────────────────────────────────
    print("[layer4] inferring variable identities")
    analyst = LLMAnalyst(get_default_client(), get_reasoning_client(), audit_logger)
    column_stats = column_stats_from_layer1_summary(layer1_summary_shape)
    corr_dict = correlations_from_df(pooled_corr["pearson"])

    identities = analyst.infer_column_identities(columns, column_stats, corr_dict, clusters, causal)
    print("[layer4] classifying measured vs. manipulated variables")
    var_types = analyst.infer_variable_type(columns, column_stats, causal, identities)
    print("[layer4] proposing cluster role hypotheses")
    cluster_roles = analyst.propose_cluster_roles(clusters, identities, column_stats)

    exporter.export_layer4_llm({
        "column_identities": identities,
        "variable_types": var_types,
        "cluster_roles": cluster_roles,
    })

    # ── Report + figures ─────────────────────────────────────────────
    print("[report] writing report.md")
    exporter.write_report(
        metadata=dataset_meta,
        pooled_correlations=pooled_corr,
        clusters=clusters,
        causal_aggregated=causal,
    )

    print("[plotting] rendering figures")
    plotter = PipelinePlotter(os.path.join(exporter.output_dir, "figures"))
    plotter.generate_all(
        runs=runs,
        columns=columns,
        pooled_correlations=pooled_corr,
        clusters=clusters,
        causal_aggregated=causal,
    )

    build_info = {
        "n_runs": len(run_ids),
        "run_ids": [str(r) for r in run_ids],
        "total_runs_in_dataset": dataset_meta.get("n_runs", len(run_ids)),
        "include_causal": True,
    }
    with open(os.path.join(exporter.output_dir, BASELINE_DONE_MARKER), "w", encoding="utf-8") as f:
        json.dump(build_info, f, indent=2)

    print(f"[done] {args.dataset_name} is now ready in the dashboard -> {exporter.output_dir}")


if __name__ == "__main__":
    main()
