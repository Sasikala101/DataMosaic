# AutoClean

An independently implemented data preprocessing platform matching the reference app's three-step **Setup → Live Agents → Results & Reports** experience. Built with **React, FastAPI, LangGraph, Groq, Pandas, NumPy**, and scikit-learn.

See [reference parity notes](docs-reference-parity.md) for inspected behavior, verified features, and intentional differences.

## Run locally

Python 3.11+ and Node 18+ are required (Python 3.12 and Node 22 recommended).

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.lock.txt
npm ci --prefix frontend
npm run build --prefix frontend
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. For frontend development, run `npm run dev --prefix frontend` in another terminal; Vite proxies `/api` to port 8000. Use one backend worker; the embedded background executor supports two simultaneous pipelines. The Python lockfile records the verified versions; `requirements.txt` lists the direct supported ranges.

## Workflow

1. Select a CSV, drag/drop it, load the synthetic sample, or provide a direct public HTTPS CSV URL.
2. Choose general ML, classification, regression, or clustering and optionally provide preprocessing instructions and a target column.
3. Launch once. Follow eight agent cards, decisions, actual statistics and a live console.
4. Review the quality score, before/after chart, preview, and executive report.
5. Download cleaned CSV, HTML executive report, a standalone Python script, Jupyter notebook, or full audit JSON.
6. Ask the dataset assistant about the profile and executed changes.

The active job ID is saved in browser session storage. Reloading restores its dataset, agent events, and results. Files and events persist under `backend/data`. If the server restarts during a run, that job becomes failed with a clear retry message; partially running work is not silently resumed.

## Agent architecture

`START → Orchestrator → Profiler → Imputation → Outlier → Encoder → Transformer → Reduction → Sampler → Validation → END`

The orchestrator chooses a bounded plan from the objective, profile and instructions. Agent nodes record `running`, `completed`, or `skipped` events. Unselected operations never transform the data. The profiler stage also executes scheduled whitespace normalization and exact deduplication. Validation always runs.

Groq plans and explains when enabled; deterministic Pandas/scikit-learn tools perform transformations. The workflow does not execute model-generated code. The exported script freezes the executed choices and includes the actual transformation functions, so it runs independently without Groq or LangGraph. It recomputes fitted statistics on its input; it is not a persisted fitted inference pipeline.

### Operations

- Imputation: adaptive mean for low skew, median for high skew; categorical mode. Entirely empty columns stay missing.
- Outliers: IQR detection; adaptive log1p for nonnegative right-skewed fields, otherwise clipping. Explicit clip/log preferences are available. Log skips negative fields.
- Encoding: one-hot for 2–20 categories, bounded to 300 output columns. Mapping recorded in audit.
- Transformation: standard or min-max scaling; clustering enables standard scaling by default.
- Reduction: standardized PCA retaining 95% variance; automatically considered for clustering with more than 20 numeric features.
- Sampling: reproducible random sampling; classification targets with imbalance above 2:1 can use random oversampling. Intentional repeated rows are reported.
- Target and IDs are excluded from feature transforms. IDs named `id`, `index`, `uuid`, or ending in `_id` can be dropped by explicit request. Row deletion/sampling affects entire rows. Target inference only recognizes conventional names and is explained in the orchestrator card.

## Custom instructions

Local mode accepts explicit **whole-dataset** instructions, separated by semicolons:

```text
use median; clip outliers; standard scaling; skip PCA
min-max scaling; sample 500 rows; keep IDs
balance classes
preserve rows; skip encoding
```

Unsupported clauses fail with a useful error instead of being silently ignored. Per-column transformations are not currently supported. A Groq key enables broader phrasing within the same bounded operation/preference schema.

## Groq configuration

Copy `.env.example` to `.env`, set `GROQ_API_KEY` locally, restart the server, and enable **Use Groq AI assistance** under Target column & AI settings. Do not commit the key. The default model is `openai/gpt-oss-120b`; set `GROQ_MODEL` for your account if necessary.

Only when enabled, Groq receives column names, aggregate statistics, custom instructions, questions, and execution audit details. Raw preview rows are excluded; audit details can still include category labels. The key remains server-side. If AI planning fails, local objective rules and the explicit instruction grammar are used and reported. Live Groq calls have not been verified without a key.

## Import limits and URL handling

- UTF-8, comma-delimited CSV, at most 100 MB, 250,000 rows, 300 columns, and 5 million input cells.
- URL imports require public HTTPS on port 443, with no URL credentials. Redirect destinations are validated individually. Connections pin the validated public IP while retaining hostname certificate/SNI checks, blocking private/local address access and DNS rebinding. Downloads have size/time bounds.
- Pandas infers types; numeric-looking fields may lose leading zeros. Nonfinite numeric values become missing on import. The preserved original is the parsed dataset, not a byte-for-byte archive.
- CSV export escapes formula-like text for spreadsheet safety.

## Validation

```sh
.venv/bin/python -m pytest backend/tests -q
npm run build --prefix frontend
```

Coverage includes the legacy manual API, autonomous pipelines, all eight agent events, replay, failure states, protected target/IDs, no-op runs, encoding collisions, PCA, balancing, input validation, SSRF/redirect checks, script reproduction, notebook compilation, and all artifact downloads. Browser QA covers the sample pipeline, agent cards, reports, chat, and reload recovery.

## Deployment

The Dockerfile serves React and the API from the same origin. `render.yaml` prepares a Render service with a **paid Starter plan and persistent disk**. No public deployment is created by these files.

```sh
docker build -t autoclean .
docker run --rm -p 8000:8000 --env-file .env -v autoclean-data:/app/backend/data autoclean
```

## Practical limits

This is an exploratory preparation tool, not a leakage-safe ML evaluation pipeline. Split data first, fit preprocessors only on training data, and persist fitted transformations for inference. Oversample only the training partition. The displayed quality score is a documented heuristic, not proof of model readiness.

This local/private version has no authentication, per-user isolation, automated retention, or distributed workers. Add those before exposing it as a shared service. UUIDs identify datasets/jobs but do not replace authorization. High-cardinality text and all-null columns can remain unprocessed; review the report.
