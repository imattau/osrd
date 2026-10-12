# OSRD AI-Native Network Operations Fork — Technical Design

**Status:** proposal for a research-to-operations fork  
**Repository baseline inspected:** OpenRailAssociation/osrd `dev` branch and OSRD technical documentation, 13 October 2026  
**Primary design choice:** preserve OSRD’s railway-domain model, deterministic core, and Editoast API. Add a typed operations intelligence service that predicts operational outcomes and proposes constrained timetable changes; accept a proposal only after OSRD pathfinding, simulation, and conflict detection validate it.

## 1. Product boundary

The fork adds decision support for dispatch and timetable recovery over a network represented in OSRD. Initial scope is **plan-level network operations**: rank recovery options after a delay/closure, forecast knock-on delay, and produce conflict-free timetable variants. It does not issue real-time movement authority, control interlocking, or replace a railway safety system.

The system answers questions such as:

- Which trains and infrastructure resources will be affected by this incident?
- Which small set of changes (departure shift, stop extension/reduction, path alternative where supported, or priority change) restores the most service with the least passenger/freight impact?
- What is the calibrated uncertainty, feasibility evidence, and reason for each recommendation?

OSRD remains authoritative for infrastructure validity, train physics, route/path feasibility, signaling interactions, and conflicts. The ML system ranks candidates and estimates outcomes; a deterministic optimizer constructs candidates; OSRD validates every candidate.

## 2. Current OSRD architecture and integration facts

OSRD describes itself as a multi-service system. The key repository components are:

- **`core/`**: Rust railway-domain computation: pathfinding, train running-time simulation, multi-train simulation/conflict computation.
- **`editoast/`**: Rust application/data service. It accesses the database, exposes HTTP endpoints/OpenAPI, performs infrastructure edition, schedules core work, and manages infrastructure cache-dependent operations. Some cache-dependent endpoints are isolated in `editoast-stateful`; most are scalable/stateless.
- **`gateway/`**: externally facing authentication/routing layer (OIDC, bearer token, and development mock are documented).
- **`front/`**: React/TypeScript web client; API types are generated from Editoast OpenAPI.
- **`osrd_schemas` / generated schema packages**: RailJSON and API schema definitions; repo documentation identifies Python `osrd_schemas` as RailJSON format definition. Keep generated contracts authoritative rather than maintaining an independent parallel copy.
- **`osrdyne`**: execution drivers for Kubernetes, Docker, and process-compose.
- PostgreSQL/PostGIS and Valkey support persisted application data and caches in the service architecture.

RailJSON models network topology with track sections, nodes/switches, track-relative objects and ranges; operational points, speed sections, slopes/curves, electrification, detectors, routes and signals describe operation and physics. Rolling stock supplies traction/braking/performance and electrical characteristics. Timetable v2 makes train schedules self-contained: schedule inputs include path waypoints, timing/stops, rolling-stock reference, margins, speed-limit tag and related options. Path and simulation results are recomputable caches, not canonical timetable input. The documented timetable API includes schedule CRUD/batch creation, pathfinding/path properties, per-schedule simulation and summaries, timetable conflicts, and path projection. Exact endpoint names and request bodies must be generated/read from the pinned fork revision’s Editoast OpenAPI; published design docs can lag implementation.

**Version caveat:** the published Train Simulation v3 design says it is pending implementation and may need adjustment. Do not make phase 1 depend on its driver behavior modules. Integrate first through the current Editoast scheduling/simulation/conflict APIs. Any later in-core extension requires a reviewed stable Rust interface and independent validation.

## 3. Proposed runtime architecture

```text
Operator / Frontend
  | incident + scope + policy constraints
  v
Editoast (authz, canonical data, snapshot/version IDs)
  | signed/versioned planning request (async job)
  v
ops-intelligence service (Python initially; gRPC/HTTP internal contract)
  |---- feature builder + graph/temporal model inference
  |---- candidate generator / constrained optimizer
  |---- uncertainty and explanation bundle
  v
Editoast orchestration adapter
  | candidate schedule mutations in isolated scenario/snapshot
  | OSRD pathfinding -> simulation -> conflicts -> summaries
  v
Validated ranked plans -> Editoast API -> Frontend review/acceptance
```

### Service responsibilities

1. **Editoast adapter (Rust)**: new authenticated endpoints, authorization, immutable snapshot reads, job lifecycle, idempotency, audit records, version checks, and candidate application only into a draft scenario/timetable. Do not let the ML container access PostgreSQL directly.
2. **Ops Intelligence service (Python)**: feature extraction, model serving, candidate scoring, constrained search, calibration, and evidence bundle. Stateless inference workers; training and feature backfills are offline jobs.
3. **OSRD core**: remains the feasibility oracle and source of railway physical/signal semantics. Invoke via existing Editoast orchestration/core-task mechanisms, not bespoke duplicated simulator code.
4. **Frontend**: incident scope and policy controls; ranked alternatives, predicted impact intervals, impacted services/resources, and simulator validation state. Require explicit human apply action; support compare/undo.
5. **Storage**: canonical operations requests, model registry/artifact metadata and audit summaries in existing persistence layer where appropriate; large immutable feature/training snapshots and model artifacts in object storage. Valkey may cache feature/inference work but never be authoritative. Add no online feature store in MVP.

### Proposed API surface (new fork contract)

Add versioned routes under an operations namespace (exact prefix aligned to current Editoast conventions):

- `POST /operations/plans`: `{scenario_id, infra_id, timetable_id, incident, horizon, policy, idempotency_key}` -> `{job_id, snapshot_id}`.
- `GET /operations/plans/{job_id}` -> lifecycle, status, snapshot/model versions, ranked plans, metrics, uncertainty, validation checks, explanation references.
- `POST /operations/plans/{job_id}/apply`: `{plan_id, expected_snapshot_id}` -> creates a draft timetable revision only if optimistic concurrency check passes.
- `POST /operations/feedback`: user disposition/reason and (where available) realized outcome, linked to plan and snapshot.

Incident schema is explicit and typed: `kind` (delay, track closure, speed restriction, rolling-stock unavailability), affected RailJSON resource IDs/ranges, start/end times, severity/source, confidence, and provenance. Do not accept free-text as the machine contract. Human notes may be stored separately and not used as model input by default.

Planning request is immutable and binds to: `infra_id + infra_version`, timetable/scenario revision, rolling-stock versions, OSRD build/schema version, incident revision, policy version, feature-builder version and model version. Reject stale apply requests and replan if any binding has changed.

## 4. Non-LLM model architecture

### 4.1 MVP: forecast + learned candidate ranker

Build an **operations graph-temporal model** over OSRD’s graph rather than a language model:

- Graph nodes: operational points, junctions, and optional aggregated resource zones.
- Edges: directed traversable track/resource segments with distance, gradient/curve summaries, speed profile, electrification/signaling/resource attributes, and route alternatives.
- Train events: departure/arrival/pass/stop events with schedule, simulated or observed time, train class/rolling-stock features, and current delay state.
- Resource occupancy: time-windowed occupation/reservation/capacity features generated from OSRD path/simulation/conflict outputs and, where lawfully available, operational feeds.

**Encoder:** 3–4 layer relational graph neural network (message passing with edge-type embeddings) over the corridor/network subgraph, joined with a temporal event encoder (causal Transformer or temporal convolution over the previous 2–6 hours; choose TCN for simpler bounded inference in first production candidate). The graph encoder captures alternate paths and propagation; train-event encoder captures headways, dwell, rolling-stock and time-of-day patterns.

**Heads:** (a) quantile regression for arrival/departure delay at each operational point, p10/p50/p90 and calibrated interval; (b) event classification for missed connection, cancellation, resource conflict, and recovery feasibility; (c) candidate-plan utility/ranking head for observed/simulated candidate outcomes; (d) optional counterfactual delta head estimating change versus baseline. The delta head is only a cheap search heuristic. It never certifies feasibility.

**Baseline comparisons:** gradient-boosted trees over engineered event/resource features; persistence/linear delay propagation; OSRD-only rule baseline. A GNN is retained only if it demonstrates robust gains over these simpler baselines on held-out corridors/time periods.

### 4.2 Candidate generation and optimization

Use a two-stage constrained optimizer:

1. Generate bounded candidate edits using deterministic domain rules and a beam search/MILP where variables are discrete and sizes remain tractable. MVP edits: departure shifts in a bounded interval, allowed dwell/stop adjustments, priority swaps within policy, and only path alternatives already representable by OSRD’s pathfinding inputs. Exclude train cancellation and rolling-stock swaps until business semantics and data exist.
2. Score/prune candidates with the learned delay/risk ranker; run final candidates through OSRD pathfinding/simulation/conflict checks.

Objective is policy-parameterized, for example minimize weighted passenger delay + freight delay + cancellations (later) + number/magnitude of edits + uncertainty penalty, subject to hard constraints: valid paths, no newly introduced OSRD conflicts for an accepted plan, schedule bounds, operational minimum dwell/headway, resource closures, rolling-stock compatibility, and user-specified protected services. Model weights cannot relax hard constraints.

Use robust optimization: optimize against p90 delay or scenario ensemble for critical services, not only mean predictions. Return Pareto alternatives (service recovery vs edit magnitude vs confidence), not a single opaque “best” solution. OSRD validator must evaluate candidate as a whole timetable because pairwise/local predictions can miss cascading interactions.

### 4.3 Later model extensions

- Multi-agent or differentiable surrogate models only for search acceleration after sufficient validated data.
- Learned train-driving policy or driver behavior module is out of scope until a separately justified task, stable core API, domain experts, safety case, and stronger assurance work exist. If explored, it may propose driving instructions in offline simulation; OSRD constraints and an independent conservative validator retain veto authority.
- No LLM in decision path. Optional natural-language incident intake may be developed separately, but must compile into a user-confirmed typed incident object and cannot write operations plans directly.

## 5. Training-data pipeline

### Sources

1. **Synthetic OSRD simulation data (starting point):** generated from RailJSON infrastructure and rolling stock, timetable schedules, pathfinding results, standalone and multi-train simulations, conflict outputs, and parameterized perturbations (late departure, dwell variance, closure, speed restriction, degraded capacity). OSRD’s realistic representative rolling stocks and test infrastructures enable reproducible initial datasets, but toy/test data is not evidence of real-world generalization.
2. **Historical operational data (needed for real operations):** planned and actual train event times, cancellations, train identity/class, rolling-stock assignment where permitted, resource occupations, incidents/temporary restrictions, and service priority/connection impact. Each operator feed needs a source adapter, legal basis, data-sharing agreement, data-quality profile, timestamp/timezone normalization, and retention policy.
3. **Human decision data:** candidate options shown, accepted/rejected edits and reason codes, overrides, and realized outcome. Rejections are not automatically “bad labels”; they may reflect constraints not present in model features. Treat as preference evidence with provenance.

### Pipeline stages

- **Ingest:** append-only raw zone; schema version and source provenance. Keep personal data out; pseudonymize train/service identifiers where operationally useful.
- **Normalize:** UTC event times plus original timezone, stable IDs mapped to versioned OSRD infrastructure resources, meters/seconds/SI units, missingness and quality flags; preserve planned vs observed vs simulated distinction.
- **Snapshot:** build immutable scenario examples binding infra, timetable, rolling-stock, OSRD/core version, incident, and input-data watermarks.
- **Replay/synthesize:** invoke OSRD batch interfaces to produce path, speed/time curves, summaries, conflicts and resource occupancy; generate perturbation episodes from seeded scenario templates.
- **Label:** construct targets by joining predicted target windows to actual/simulated event outcomes (details below); preserve label source and confidence.
- **Split:** temporal holdout and corridor/infra holdout; never randomly split adjacent events from the same disruption across train/test. Include unseen rolling stock and signaling systems as explicit generalization slices.
- **Publish:** signed dataset manifest (source, license, snapshot IDs, schemas, code commit, filtering, label definitions, split hashes), Parquet files, feature stats; reproducible transform and training runs.
- **Govern:** data minimization, operator consent/authorization, deletion workflow, access logging, retention and model-training opt-out at source agreement level.

### Labels and targets

| Target | Definition | Source |
|---|---|---|
| Arrival/departure delay | Actual minus planned time at each stop/pass event; seconds, signed | Operational actuals; simulated analogue for synthetic pretraining |
| Delay quantiles | Conditional p10/p50/p90 delay over horizon and event point | Historical replay/forecast origin |
| Delay propagation | Incremental delay attributable after a disruption/resource interaction; label as predictive association unless causal design supports attribution | Replay with incident and matched baseline; simulation paired scenarios |
| Event outcome | on-time bucket, missed connection, cancellation, not-run/unknown (distinct classes) | Operational feed |
| Conflict/feasibility | OSRD result: path/simulation valid, conflict count/types, failed reason; never infer from observed actual alone | OSRD core |
| Recovery-plan utility | realized weighted service impact and edits for an applied plan; synthetic paired outcomes for simulation-only | Actual replay / OSRD validation |
| Human preference | accepted, rejected, edited, with standardized reason and policy context | Operator UI feedback |
| Incident state | affected resources, interval, kind, severity and source confidence | Versioned incident source |

For causal recovery-effect claims, use paired OSRD counterfactual simulation as primary label and, after deployment, matched observational analysis with confounding caveats. Do not call raw before/after delay change “causal impact.”

## 6. Evaluation and release gates

### Offline ML metrics

- Arrival delay: MAE and pinball loss at p10/p50/p90; calibration error/coverage for 80% and 90% intervals.
- Propagation: MAE by horizon, incident class, distance, traffic density, corridor, signaling system, rolling-stock class.
- Event outcomes: PR-AUC/recall at fixed alert budget for rare cancellations/missed connections; Brier score and calibration.
- Ranker: top-k regret against best OSRD-validated candidate in the enumerated candidate set; pairwise preference accuracy only as a secondary measure.
- Coverage and abstention: percent of scenarios meeting applicability/confidence thresholds; explicit out-of-distribution detection performance.

### System/operations metrics

- Candidate OSRD validation pass rate; newly introduced conflict rate must be zero for plans offered as “validated.”
- Improvement in weighted delay/recovery objective vs current baseline, with confidence intervals and stratification.
- Planning latency p50/p95, simulation calls per job, candidate count, cache hit rate, service error rate.
- Human acceptance, edit distance from proposal, override rate, and outcome after application.

### Evaluation design

Temporal backtest with rolling origins; entire corridors/infrastructures held out; incident-type and peak/off-peak slices; stress test extreme but physically possible disruption combinations. Compare rule baseline, OSRD optimizer without ML, ranker with OSRD validation, and operator historical choices. Run paired seeds for synthetic simulation. Detect train/test contamination by shared timetable/scenario IDs. Publish confidence intervals and data coverage with each release.

### Rollout gates

1. **Research:** beat simple baselines on temporal holdout and show calibrated uncertainty; no claim of operations readiness.
2. **Shadow:** live read-only predictions; compare to realized outcomes; no recommendations shown as actionable.
3. **Decision support:** show ranked alternatives only when input quality/applicability gates pass; human reviews and applies draft timetable; all candidates revalidated by OSRD.
4. **Controlled operations pilot:** operator-specific assurance, documented procedures, rollback, monitoring and independent safety review. No autonomous application in this design.

## 7. Safety assurance and security

- **Safety boundary:** advisory planning only; explicitly prohibit movement authority, signal/interlocking commands, and direct safety-critical actuation in API, IAM, network policy and documentation.
- **Hard validation:** every surfaced plan must pass current OSRD validation on the exact bound infra/timetable snapshot; any unknown validation status means not validated. If simulation returns invalid/partial/timeout, show as unvalidated and do not offer one-click application.
- **Fail-safe behavior:** model down, stale inputs, distribution shift, missing features, low confidence, or validator failure -> no AI recommendation; existing OSRD manual workflow remains available.
- **Human control:** operator chooses plan, sees changed fields, predicted impact intervals, affected services/resources, validation evidence and explanation; explicit confirmation applies only to draft scenario with revision check and undo.
- **Conservative uncertainty:** calibrated intervals and OOD checks; abstain outside training support. Validate by infrastructure, signaling system, class, disruption type, weather where available, and traffic density. Avoid claims that explanations are causal unless established.
- **Independent assurance:** hazard analysis, misuse cases, safety/security review, data governance review, and applicable operator/regulatory standards determined before a live pilot. ML metrics alone do not establish railway safety.
- **Audit:** immutable plan record with user, timestamp, snapshot bindings, model/data/code versions, input feature digest, candidate edits, OSRD validation outputs, confidence, policy, user action and eventual outcome.
- **Security:** internal service identity, least privilege, gateway authz parity, tenant/operator isolation, signed model artifacts, pinned dependencies, no raw database credentials in model container, request rate/size bounds, data egress controls.

## 8. Deployment and operations

- Local development: add `ops-intelligence` to Docker Compose/process-compose with deterministic fake/small_infra examples and seeded model artifact; optional inference disabled by default.
- Production: deploy inference workers alongside existing OSRD services under Osrdyne/Kubernetes; gateway reaches Editoast only; Editoast dispatches jobs via existing asynchronous/core task patterns. Scale stateless workers separately from stateful infra-cache service. Set resource quotas, concurrency limits and timeouts for OSRD simulation fan-out.
- Training: offline isolated job; no GPU required for the initial small graph/TCN model, though accelerators may speed retraining. Do not require GPU at runtime unless benchmarked.
- Artifacts: registry stores model binary, feature schema, training manifest, code commit, evaluation report, approval state and signatures. Canary/shadow model versions; one config switch to rollback. Pin model and feature schema in every job.
- Observability: OpenTelemetry traces across Editoast job, inference, candidate generation and OSRD validation; metrics for quality, latency, abstention, drift, validation failures and user override. Do not log raw operator feeds or sensitive payloads.
- Data/model drift: monitor feature distributions and residual/calibration on matured outcomes; alert and degrade to ranker-off/rules-only behavior; retraining is reviewed and gated, never automatic promotion.

## 9. Repository fork structure

```text
osrd-ai/
  core/                         # upstream OSRD core; preserve deterministic semantics
  editoast/                     # auth, snapshot pinning, job and plan endpoints
  front/                        # planning/review UI
  gateway/                      # upstream auth/routing
  osrd_schemas/                 # upstream RailJSON/API schemas
  ops-intelligence/
    api/                        # OpenAPI/protobuf internal contracts
    service/                    # inference, ranking, orchestration client
    features/                   # versioned feature transforms
    models/                     # graph-temporal model and baselines
    optimizer/                  # candidate generation and constrained objectives
    training/                   # replay, labels, splits, training/eval
    registry/                   # model manifest and promotion metadata
    tests/                      # unit, contract, golden and simulation tests
  ops-data/                     # synthetic generators, source adapters, manifests (no licensed raw data)
  docs/
    architecture/ai-operations.md
    data-governance.md
    safety-case/assumptions-and-hazards.md
    runbooks/
  tests/ai_operations/          # OSRD end-to-end fixture scenarios
```

Forking practice: track upstream `dev` and keep AI functionality mostly additive. Prefer upstream-compatible schema/API extension and minimal core patches. Maintain an upstream merge log and run OSRD’s existing CI plus AI contract/golden suites on every merge. Do not vendor a divergent RailJSON schema.

## 10. Milestones and acceptance criteria

| Phase | Duration (indicative) | Deliverable / exit criteria |
|---|---:|---|
| 0. Baseline and fork | 2–3 weeks | Pin upstream commit; build full OSRD stack; document actual OpenAPI and core-task flows; threat/hazard boundary; reproducible small_infra scenarios |
| 1. Dataset and simulator replay | 4–6 weeks | Versioned RailJSON/timetable snapshot pipeline; perturbation generator; OSRD outputs and labels; leakage-safe splits; baseline dashboard |
| 2. Forecast baseline | 4–6 weeks | Delay quantile baseline (GBDT/TCN), calibration, corridor/time holdouts, applicability gate; no UI action path |
| 3. Planning API and optimizer | 6–8 weeks | Typed incident and policy contract; candidate generation; OSRD validation orchestration; immutable job record; pass-through rules baseline |
| 4. Learned ranking + UI | 6–8 weeks | Ranker improves held-out top-k regret; compare plans with uncertainty/evidence; explicit draft apply/undo; authz/audit tests |
| 5. Shadow pilot | 8–12 weeks | Operator feed agreement and source adapter; live shadow forecasts, drift/latency dashboards, incident review; no operational actuation |
| 6. Controlled decision-support pilot | 3–6 months | Operator-specific assurance case/procedures; measured improvement, acceptable overrides and no invalid surfaced plans; staged approval |

Timelines assume a small cross-functional team (rail operations/domain expert, OSRD Rust engineer, ML/data engineer, frontend engineer, safety/security review). They are planning estimates, not commitments; access to real operational data is the largest schedule dependency.

## 11. First implementation slice

The first code change should be an **offline replay prototype**, not a new model serving endpoint:

1. Pin an OSRD `dev` SHA and export current Editoast OpenAPI/schema artifacts.
2. Generate a small set of seeded timetables on `small_infra`, perturb departure times/dwell and closures, and run current pathfinding/simulation/conflicts.
3. Write versioned Parquet episode records with exact inputs, outputs, labels, and manifest.
4. Establish simple delay propagation and GBDT baselines with temporal/scenario splits.
5. Only after data contract and labels are stable, add the advisory planning API and UI.

This ordering tests the core hypothesis—whether OSRD plus available operations data can support useful, generalizable predictions—before creating a production service boundary.

## 12. Assumptions, risks, and unresolved decisions

- Public documentation is not a substitute for checking the exact chosen `dev` SHA. Editoast endpoints and DTOs evolve; regenerate API clients and schema bindings from that checkout.
- Synthetic OSRD runs provide internally consistent labels but will not by themselves teach real delay causes, dispatch preferences, or feed quality. An operator data partner is required for operational validation.
- OSRD conflict detection’s documented design aims to avoid false negatives relative to the corresponding multi-train simulation; nevertheless this fork must treat current implementation outputs as versioned validation evidence, not a proof of real-world safety.
- Infrastructure and train identity licensing/ownership, actuals retention, cross-operator transfer, and privacy rules must be settled for each data source.
- Optimization objective weights and protected-service rules are operator policy, not model-learned defaults.
- Path changes may have constraints not yet represented in generic train-schedule edits; only expose candidate actions that current API/core can validate faithfully.
- Applicable railway assurance requirements depend on jurisdiction and whether recommendations are used in live operations; determine with the operator before pilot.

## Primary references inspected

- [OSRD repository](https://github.com/OpenRailAssociation/osrd)
- [OSRD contribution guide and component map](https://github.com/OpenRailAssociation/osrd/blob/dev/CONTRIBUTING.md)
- [OSRD services architecture](https://osrd.fr/en/docs/reference/architecture/services/)
- [OSRD timetable v2 design and endpoints](https://osrd.fr/en/docs/reference/design-docs/timetable/)
- [OSRD train simulation v3 status](https://osrd.fr/en/docs/reference/design-docs/train-sim-v3/)
- [OSRD driving instructions](https://osrd.fr/en/docs/reference/design-docs/train-sim-v3/driving-instruction/)
- [OSRD driver behavior modules](https://osrd.fr/en/docs/reference/design-docs/train-sim-v3/driver-behavior-modules/)
- [OSRD conflict detection](https://osrd.fr/en/docs/reference/design-docs/conflict-detection/)
- [OSRD infrastructure model example](https://osrd.fr/en/docs/explanation/models/data-models-full-example/)
- [OSRD running-time calculation](https://osrd.fr/en/docs/explanation/running_time_calculation/)
- [OSRD Editoast API docs](https://osrd.fr/en/docs/reference/api/editoast/)
