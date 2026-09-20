# TPM Pipeline

A general-purpose statistical analysis pipeline for **numeric, multi-run tabular data**.

TPM takes raw datasets, builds a statistical baseline, detects changes in new data, and uses an LLM to help explain what the analysis may mean.

**[Watch the demo](https://youtu.be/RfT0fCZjg3s)**

## What it does

| Layer | Purpose | Main output |
|---|---|---|
| **0 — Ingestion** | Load, clean, and detect dataset structure | Normalized data, run/sample/label columns |
| **1 — Profiling** | Build a statistical fingerprint | Statistics, correlations, quality alerts |
| **2 — Relational / Causal** | Find relationships and causal structure | Correlations, clusters, causal links |
| **3 — Drift & Anomaly** | Compare new runs with a normal baseline | Anomaly scores, drift details, fault onset |
| **4 — LLM Interpretation** | Explain the statistical results | Variable identities, cluster roles, root-cause hypotheses, chat |

> **Important:** Layers 0–3 produce deterministic statistical results. Layer 4 produces LLM-generated hypotheses and should be treated as interpretation, not ground truth.

## Key Features

- **Automatic dataset discovery** — detects delimiters, timestamps, labels, run IDs, sample IDs, and numeric columns.
- **Statistical profiling** — uses `ydata-profiling` for per-run statistics, correlations, time-series diagnostics, and quality alerts.
- **Causal discovery** — uses Tigramite PCMCI with partial correlation tests.
- **Automatic clustering** — groups related variables using hierarchical clustering and silhouette-based cluster selection.
- **Multi-perspective anomaly detection** — checks mean shifts, variance shifts, correlation changes, and fault onset.
- **Autocorrelation-aware statistics** — adjusts effective sample size for sequential data.
- **CUSUM fault detection** — identifies when changes begin and provides a possible propagation order.
- **LLM-assisted interpretation** — explains variables, cluster roles, and possible root causes from statistical evidence; report-grounded chat and an on-demand "explain this graph" for any chart.
- **LLM tool-calling** — the chat model can call a tool on its own judgment (e.g. drafting a data-quality rule from a plain-English message); adding another tool is a JSON schema plus one handler, no architecture changes.
- **Data-quality rules** — one built-in missing-data check plus human-confirmed custom rules authored through chat; applying a saved rule is deterministic, no LLM involved at check time.
- **Audit logging & flags** — every LLM call and flagged data-quality decision is recorded and browsable in the dashboard's Flags tab.
- **Reusable reports and plots** — exports structured JSON/CSV data plus a readable `report.md`.
- **Web dashboard** — inspect results and upload new runs through a FastAPI/Jinja2 interface.
- **Dataset-agnostic design** — not tied to the original Tennessee Eastman dataset.

## Typical Workflow

```text
Raw dataset
    │
    ▼
Layer 0 ── Ingestion & normalization
    │
    ▼
Layer 1 ── Statistical profiling
    │
    ▼
Layer 2 ── Correlation, clustering & causal analysis
    │
    ▼
Baseline
    │
    ├──────────────► Dashboard / reports
    │
    ▼
New run
    │
    ▼
Layer 3 ── Drift & anomaly detection
    │
    ▼
Layer 4 ── Root-cause hypotheses & explanation
```

## Quick Start

From the project root:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your `OPENAI_API_KEY`.

Start the web interface:

```bash
python -m uvicorn tpm.web.app:app --reload
```

Then open:

```text
http://127.0.0.1:8000/
```

## Web Dashboard

1. Select a dataset.
2. Build a baseline from a sample of runs.
3. Explore statistics, plots, correlations, clusters, and causal relationships — click **Explain this graph** on any chart for a plain-language summary grounded in that chart's own data.
4. Teach the system a data-quality rule in plain English through chat (e.g. "this variable should stay between 0 and 100") — it drafts a rule for you to confirm before saving.
5. Upload a new run with **Check New Data**.
6. Review drift/anomaly results — root-cause hypotheses only appear when the run is actually anomalous enough to warrant one.
7. Check the **Flags** tab for anything raised for human review.

Baseline analysis can take several minutes because causal discovery and LLM analysis are computationally expensive.

## Outputs

Results are stored per dataset:

```text
analysis_output/<dataset_name>/
```

Important outputs include:

- `report.md` — combined human-readable report.
- `dataset_meta.json` — detected dataset structure.
- `layer1/` — per-run profiling, pooled variable stats, and data-quality results.
- `layer2/` — correlations, clusters, and causal-discovery results.
- `layer4/` — LLM interpretation results (variable identities, types, cluster roles).
- `figures/` — every dashboard plot (histograms, traces, correlation heatmap, cluster dendrogram, causal graph).
- `quality_rules.json` — saved custom data-quality rules.
- `run_data/` — reusable per-run data, so plots can be regenerated without rerunning Layers 0-2.
- `audit_log.jsonl` — every LLM call and flagged data-quality decision.

Drift/anomaly results from **Check New Data** (Layer 3) are computed and
shown live but not persisted to disk — each check reloads the baseline
fresh, so nothing needs rerunning if the baseline changes.

Regenerate plots without rerunning the pipeline:

```bash
python -m tpm.utils.plotting analysis_output/<dataset_name>
```

## LLM Provider & Tools

Layer 4 talks to any backend that speaks OpenAI's Chat Completions
protocol — the real OpenAI API, or a self-hosted server (Ollama, vLLM,
LM Studio, llama.cpp) via a `base_url` override. Swapping the provider,
model, or pointing at a local model instead of the API is an
environment-variable change, not a code change:

- `LLM_MODEL` / `LLM_REASONING_MODEL` — route simple tasks (variable
  identity, variable type) to a cheap model and harder reasoning
  (root-cause analysis) to a stronger one, independently.
- `LLM_PROVIDER=mock` — swap in a canned-response client for tests, no
  network or API key needed.
- `LLM_BASE_URL` / `OPENAI_API_KEY` — point at any OpenAI-compatible
  server instead of the real API.

**Tool-calling**: the dashboard chat gives the model a tool it can call
on its own judgment — right now, drafting a data-quality rule from a
plain-English message. The model decides when to use it; a human still
has to confirm before anything is saved. Adding another tool is a JSON
schema plus one handler branch in `chat()` — no architecture changes.

**Never raw data**: every Layer 4 method rejects any pandas DataFrame
passed into it — enforced by a decorator checked at call time, not just
a convention — so the LLM only ever sees derived statistics/summaries,
never raw data rows.

## Data Quality

Data quality is separate from anomaly detection:

- **Warnings** are reported but do not block analysis.
- **Critical issues** can prevent unreliable data from reaching drift/fault analysis.
- Custom range rules can be proposed through chat and must be explicitly confirmed before being saved.
- Quality-rule evaluation is deterministic and does not require an LLM.

## Design Principles

- **Dataset-agnostic** — no assumptions about specific sensors, units, or domains.
- **Statistically grounded** — prefer established statistical methods over dataset-specific magic numbers.
- **Auditable** — save the evidence and configuration behind analysis results.
- **Human-in-the-loop** — LLM output is treated as hypotheses, not fact; anything the LLM proposes that would change future behavior (a new data-quality rule) requires explicit human confirmation before it's saved.
- **Reusable** — saved results can be reloaded and visualized without recomputing the analysis.

## Project Structure

```text
tpm/
├── layer0_ingestion.py
├── layer1_profiling.py
├── layer2_relational.py
├── layer3_drift.py
├── layer4_llm.py
├── web/
│   ├── app.py
│   ├── jobs.py
│   ├── datasets.py
│   └── pipeline_service.py
└── utils/
    ├── data_quality.py
    ├── quality_rules.py
    ├── llm_client.py
    ├── audit_log.py
    ├── results_exporter.py
    └── plotting.py
```

## Original Dataset / Testing

The project was originally developed and tested with Tennessee Eastman process data, but the pipeline is designed for other numeric multi-run datasets as well.

The demo workflow includes normal and abnormal sample runs that can be uploaded through **Check New Data**.

## Performance Note

Causal discovery is the most expensive part of the pipeline. The web service uses a smaller causal-search range to keep baseline generation practical, while the fuller Layer 2 configuration remains available for offline/batch analysis.

