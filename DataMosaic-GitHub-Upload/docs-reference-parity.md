# Reference behavior and implementation parity

Reference inspected: https://autoclean-xd62.onrender.com/ui on 2026-10-06.
A synthetic customer CSV was processed in the reference UI. No private user dataset was uploaded.

| Reference surface | Local implementation | Verification |
| --- | --- | --- |
| Monochrome, rounded header with Setup / Live Agents / Results | Same layout, navigation, black/white surfaces and blue actions | Browser inspection |
| CSV file selection, drag/drop, 100 MB limit | Implemented; additionally bounded to 250k rows, 300 columns, 5M cells | Upload/parser tests, browser sample |
| Public dataset URL and launch | HTTPS fetch, redirect validation, bounded download, same pipeline | Public/private URL tests |
| General/classification/regression/clustering objective | Objective-sensitive plan, target inference, optional target override | Automated tests |
| Custom preprocessing preferences | Constrained Groq planner; explicit local instruction grammar | Tests for supported and unsupported input |
| One-click pipeline | Background LangGraph job starts after import | Browser end-to-end test |
| Eight agent flow nodes/cards | Orchestrator, Profiler, Imputation, Outlier, Encoder, Transformer, Reduction, Sampler | All 8 states checked |
| Live status and console logs | Real execution events delivered over SSE, persisted and replayable | SSE replay tests and browser inspection |
| Quality score and chart | Derived before/after metrics; disclosed score formula | Formula tests; browser results |
| CSV / report / script / notebook | Clean CSV, standalone HTML report, executable Python, Jupyter notebook | Download checks; Python reproduction; notebook compilation |
| Full report and dataset preview | Original profile, changes, skipped decisions, target, objective, all output columns | Browser inspection |
| Dataset assistant | Groq-grounded answers when enabled; factual local summaries otherwise | Browser chat and API tests |
| New Pipeline | Clears current session; keeps previous run files intact | Browser control |

## Deliberate differences and unknowns

- The reference source, system prompts, exact algorithms, score formula, and hosting configuration are not accessible through its UI. This is an independently implemented behavioral match, not a source clone or promise of identical outputs.
- The reference sample run retained duplicates and whitespace variants, dropped the ID, and encoded its inferred target. The local default trims whitespace, removes exact duplicates, and preserves IDs and target values. Dropping IDs can be requested. This keeps useful protections from the original local implementation.
- The local score uses 60% completeness, 30% unique rows, 10% numeric values within IQR bounds, with the formula displayed. It does not imitate unverified reference score calculations or claim model readiness.
- Sample processing with local rules can finish almost immediately. Events are genuine; no fake timing delays are added.
- No Groq key was supplied, so live Groq requests cannot be verified. Planning, execution and chat are labeled with their actual mode.
- This remains a local/private exploratory tool. Authentication, per-user access controls, retention policies, and a distributed worker queue are not implemented. The job runner supports a single server process with two concurrent pipelines.
