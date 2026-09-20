"""
Batch driver: Layers 0-2 across a sample of simulationRuns — local use.
------------------------------------------------------------------------
Runs Layer 0 (ingestion), Layer 1 (per-run profiling + data quality) and
Layer 2 (pooled correlations, clustering, per-run PCMCI causal discovery
aggregated across runs) over a chosen sample of runs in a multi-run CSV.

No Layer 3 (needs a baseline/test-run split) and no Layer 4 (LLM calls,
needs OPENAI_API_KEY + network) — just the deterministic statistical/
causal pipeline, matching what was asked for.

Usage (from the project root):

    python process_runs.py --filepath te_process_normal_train.csv \
        --dataset-name te_process_train_100 --n-runs 100 --seed 0

Output layout: <output-dir>/<dataset-name>/... — the same structure
ResultsExporter always writes (see tpm/utils/results_exporter.py's
docstring), rooted at --output-dir (default ./analysis_output).
"""

import argparse
import random
import time

from tpm.layer0_ingestion import DataIngestion
from tpm.layer1_profiling import StatisticalProfiler
from tpm.layer2_relational import CausalAnalyzer
from tpm.utils.results_exporter import ResultsExporter
from tpm.utils import quality_rules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filepath", required=True, help="Path to the multi-run CSV")
    parser.add_argument("--dataset-name", required=True, help="Label for this run — used as the output subfolder name")
    parser.add_argument("--output-dir", default="analysis_output", help="Base output dir (default: ./analysis_output)")
    parser.add_argument("--n-runs", type=int, default=None, help="Process a random sample of N runs instead of every run in the file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for --n-runs sampling (reproducible)")
    parser.add_argument("--tau-max", type=int, default=None, help="PCMCI max lag; omit to use CausalAnalyzer's own default (10, thorough but slow — see the timing note below)")
    args = parser.parse_args()

    print(f"[load] {args.filepath}")
    t0 = time.time()
    ingestion = DataIngestion()
    data = ingestion.load_multi_run(args.filepath)
    all_run_ids = data["run_ids"]
    columns = data["columns"]

    if args.n_runs is not None and args.n_runs < len(all_run_ids):
        rng = random.Random(args.seed)
        run_ids = sorted(rng.sample(all_run_ids, args.n_runs))
        print(f"[load] sampling {len(run_ids)} of {len(all_run_ids)} runs (seed={args.seed})")
    else:
        run_ids = all_run_ids

    runs = {rid: data["runs"][rid] for rid in run_ids}
    print(f"[load] {len(run_ids)} runs, {len(columns)} columns, {time.time() - t0:.1f}s")

    exporter = ResultsExporter(args.dataset_name, base_dir=args.output_dir)
    exporter.export_dataset_meta(data["metadata"], data["labels"])
    exporter.export_run_data(runs, columns)

    # ── Layer 1: per-run profiling + data quality ────────────────────
    profiler = StatisticalProfiler()
    custom_rules = quality_rules.load_active_rules(exporter.output_dir)
    run_descriptions = {}
    quality_by_run = {}
    for i, rid in enumerate(run_ids):
        t1 = time.time()
        rep = profiler.profile(runs[rid], columns)
        exporter.export_layer1(rid, rep)
        run_descriptions[rid] = rep.get_description()
        quality_by_run[rid] = profiler.check_quality(runs[rid], columns, custom_rules=custom_rules)
        print(f"[layer1] run {rid} ({i + 1}/{len(run_ids)}) {time.time() - t1:.1f}s", flush=True)

    exporter.export_layer1_summary(run_descriptions)
    exporter.export_layer1_quality(quality_by_run)

    # ── Layer 2: pooled correlations, clustering, per-run causal discovery ──
    print("[layer2] pooled correlations")
    kwargs = {} if args.tau_max is None else {"tau_max": args.tau_max}
    analyzer = CausalAnalyzer(**kwargs)
    pooled_corr = analyzer.compute_pooled_correlations(runs, columns)
    exporter.export_layer2_correlations(pooled_corr)

    print("[layer2] clustering")
    clusters = analyzer.cluster_columns(pooled_corr, method="spearman")
    exporter.export_layer2_clusters(clusters)

    print(f"[layer2] PCMCI causal discovery over {len(run_ids)} runs (slowest step)", flush=True)
    t2 = time.time()
    causal = analyzer.analyze_per_run(runs, columns)
    exporter.export_layer2_causal(causal)
    print(f"[layer2] causal discovery done, {time.time() - t2:.1f}s")

    print(f"[done] total {time.time() - t0:.1f}s -> {exporter.output_dir}")


if __name__ == "__main__":
    main()
