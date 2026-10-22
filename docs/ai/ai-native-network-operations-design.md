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
- If a planner requests a new recurring service, what timetable can fit it, and what bounded changes to existing services would make room for it?

OSRD remains authoritative for infrastructure validity, train physics, route/path feasibility, signaling interactions, and conflicts. The ML system ranks candidates and estimates outcomes; a deterministic optimizer constructs candidates; OSRD validates every candidate.

### Long-term workflow: plan a service and fit it into the network

The end-user capability is a planner assistant for adding or revising services. A planner
could specify a corridor or line, ordered stops (including pass-through points), days and
hours of operation, target frequency, and service/rolling-stock requirements. The system
should return a draft service timetable plus any proposed changes to existing services
needed to accommodate it. Every proposed change must be visible, bounded by planner
policy, compared with a no-change baseline, and validated as a whole timetable.

The typed request should eventually include:

- **Service pattern:** origin/destination, ordered mandatory and optional stops, pass or
  stop behavior, service days/date range, and local operating window/time zone.
- **Frequency:** target headway or trains per hour, acceptable headway variation, and
  whether the service is regular/paced or has specific departure/arrival windows.
- **Operating needs:** allowed rolling stock/service class, minimum dwell, connections or
  timed meets where supported, and any loading/platform constraints represented by OSRD.
- **Change policy:** protected existing trains, editable service set, per-service maximum
  retiming/dwell/path changes, allowed route alternatives, and the maximum number of
  existing services that may be changed.
- **Objective:** priorities such as keeping the requested frequency, minimizing passenger
  delay, minimizing existing-service edits, or protecting selected services. Hard
  railway and operator constraints remain separate from weighted objectives.

The planning job should then:

1. Resolve the request against a versioned infrastructure and timetable snapshot; report
   ambiguous stops, unsupported constraints, or unavailable rolling stock before search.
2. Expand the service pattern into candidate departures and pathfinding requests, and
   compute a whole-timetable baseline with OSRD for the selected time window.
3. Find bottlenecks and a bounded affected-service neighborhood from shared track,
   junction, platform, and time-window use. Candidate actions may adjust departures,
   dwell, or supported path choices for the new service and eligible existing services.
4. Search combinations of those actions under explicit edit bounds. A learned impact
   model may estimate network-wide delay/risk deltas and prioritize candidates, but it
   cannot mark a candidate feasible. Keep a deterministic no-ML search baseline.
5. Apply promising combinations to isolated full-timetable drafts and rerun OSRD
   pathfinding, simulation, and conflict detection. Iterate when validation rejects a
   candidate, within fixed runtime and candidate-count budgets.
6. Return several validated alternatives where possible: requested-service coverage,
   affected services and exact edits, predicted and simulated outcomes, trade-offs,
   uncertainty, validation evidence, and any unmet request constraints. The planner
   chooses and applies one draft; do not silently modify the source timetable.

The initial system should promise **bounded useful alternatives**, not a globally optimal
network timetable. It must say when it cannot satisfy the request within the permitted
edit envelope and show which constraints or conflicts prevented a solution.

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
- `POST /operations/service-plans`: `{scenario_id, infra_id, timetable_id, service_request, change_policy, objective, idempotency_key}` -> `{job_id, snapshot_id}`. `service_request` carries the typed line/stop pattern, service dates/window, frequency, and rolling-stock/service constraints.
- `GET /operations/plans/{job_id}` -> lifecycle, status, snapshot/model versions, ranked plans, metrics, uncertainty, validation checks, explanation references.
- `POST /operations/plans/{job_id}/apply`: `{plan_id, expected_snapshot_id}` -> creates a draft timetable revision only if optimistic concurrency check passes.
- `POST /operations/feedback`: user disposition/reason and (where available) realized outcome, linked to plan and snapshot.

A service-plan result must enumerate the added service and every proposed modification
to existing services as explicit before/after schedule fields. It must also distinguish
`validated`, `rejected`, `not_simulated`, and `simulation_error` candidates. Apply must
recheck the exact infrastructure/timetable revision and create a draft; it must never
silently publish schedule edits or replace the source timetable.

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

**First graph prototype:** use the published RIDE Gold Lite `gnn/` representation and
its official builder/trainer before creating a new graph encoding. It is a PyTorch
Geometric heterogeneous graph with train-node, station-node, station-link, past-event,
and known-future-event feature groups. It supplies disjoint train and 2025 test graph
shards, feature specification and normalization metadata, and a shared 15-event delay
target protocol. Gold Lite is a development-sized dataset (the full release tree is
listed as 26.5 GB); only the exact GNN metadata and train graph shards should be fetched
for initial development. Preserve its split and target alignment; run all internal
model selection within the published training period, then evaluate the official 2025
test once. The RIDE repository reports Gold Standard mean MAE of 73.62 s for GNN versus
77.20 s for MLP over ten seeds. That supports running the comparison, but is not an
OSRD-fork result and does not establish transfer to another railway or planning task.

This public GNN is event/station relational, not an OSRD infrastructure graph. Its
features do not encode RailJSON track sections, switch route choices, signaling blocks,
or validated resource occupancy. Follow it with a separate OSRD adapter: create directed
track/resource edges from a pinned infrastructure version, connect train events to
occupancy intervals and operational points, and retain typed ID mappings with provenance
and match confidence. Keep both models as delay forecasters; OSRD pathfinding,
simulation and conflict detection remain the feasibility authority.

### 4.2 Candidate generation and optimization

The service-planning task is a constrained **combinatorial search**. The model should
not generate an unconstrained timetable token by token. Use a candidate generator and
optimizer for explicit timetable variables, with ML only to estimate impacts and spend
expensive OSRD simulation calls on the most promising candidates.

For a request to add a service, represent each candidate as a typed delta over a pinned
baseline timetable:

- New-service departure times and supported route/path choices.
- Existing-service departure shifts, dwell changes, or supported path alternatives.
- Per-service edit count and magnitude, and the request constraints each edit serves.

Start by enumerating regular departure patterns from the requested headway/window and
OSRD-supported paths. Expand the affected set by shared resources and overlapping time
windows. Search bounded combinations using deterministic rules plus beam search; use
MILP/CP-SAT only where the chosen OSRD constraints can be expressed exactly. All
unmodeled or unsupported constraints must be left to OSRD validation and must not be
silently approximated as guaranteed feasibility.

The learned impact estimator for this stage must be **action-conditioned**: inputs
include the baseline timetable state, the requested new service, and the candidate delta.
Outputs estimate per-service/event delay changes and uncertainty, aggregate objective
change, and risk of OSRD validation failure. Train it on paired baseline/modified OSRD
episodes. This is distinct from the current RIDE GNN, which forecasts observed delay
without a proposed edit. The RIDE model may supply a complementary initial-delay prior;
it cannot stand in for paired counterfactual labels.

Use a two-stage constrained optimizer:

1. Generate bounded candidate edits using deterministic domain rules and a beam search/MILP where variables are discrete and sizes remain tractable. MVP edits: departure shifts in a bounded interval, allowed dwell/stop adjustments, priority swaps within policy, and only path alternatives already representable by OSRD’s pathfinding inputs. Exclude train cancellation and rolling-stock swaps until business semantics and data exist.
2. Score/prune candidates with the action-conditioned impact/risk model; run final
   candidates through OSRD pathfinding/simulation/conflict checks. Compare against the
   same bounded search without ML to measure whether the model saves validation time or
   improves the alternatives found.

Objective is policy-parameterized, for example maximize delivered requested trains and
frequency adherence, then minimize weighted passenger/freight delay, number/magnitude of
existing-service edits, and uncertainty penalty. Apply hard constraints for valid paths,
requested service window/frequency bounds, no newly introduced OSRD conflicts for an
accepted plan, schedule bounds, operational minimum dwell/headway, resource closures,
rolling-stock compatibility, and user-specified protected services. Model weights cannot
relax hard constraints. Return Pareto alternatives when objectives conflict, such as
better frequency adherence with more existing-service retiming versus fewer edits with
some requested departures unsatisfied.

Use robust optimization: optimize against p90 delay or scenario ensemble for critical services, not only mean predictions. Return Pareto alternatives (service recovery vs edit magnitude vs confidence), not a single opaque “best” solution. OSRD validator must evaluate candidate as a whole timetable because pairwise/local predictions can miss cascading interactions.

### 4.3 Later model extensions

- Multi-agent or differentiable surrogate models only for search acceleration after sufficient validated data.
- Learned train-driving policy or driver behavior module is out of scope until a separately justified task, stable core API, domain experts, safety case, and stronger assurance work exist. If explored, it may propose driving instructions in offline simulation; OSRD constraints and an independent conservative validator retain veto authority.
- No LLM in decision path. Optional natural-language incident intake may be developed separately, but must compile into a user-confirmed typed incident object and cannot write operations plans directly.

### 4.4 Model-development sequence

Develop models in stages, and keep the observed-delay forecasting task separate from
counterfactual timetable planning:

1. **Finish the RIDE GNN benchmark.** Train and evaluate it only as an observed delay
   forecaster, compare with persistence and tabular baselines, and report its temporal
   holdout results. Use it to establish whether graph structure helps this task; do not
   treat its score as evidence that it can assess proposed timetable changes.
2. **Build paired OSRD examples.** For each pinned baseline timetable, record a service
   request or explicit timetable edit, the candidate delta, and full-network outputs for
   both baseline and changed timetables. Include rejected/invalid candidates as distinct
   outcomes, not as missing examples. Split by scenario, infrastructure, and time so
   related candidates from one baseline cannot leak across train and evaluation sets.
3. **Set non-neural baselines.** Measure persistence/OSRD-only estimates, a GBDT impact
   model over typed timetable/resource features, and the deterministic search without ML.
   These establish whether a more complex model adds enough value to justify its cost.
4. **Train an action-conditioned impact model.** Input the baseline network state, new
   service specification, and exact candidate edits. Predict per-service/event delay
   deltas with uncertainty, aggregate objective change, and validation-risk indicators.
   Start with the GBDT; compare an OSRD-native relational graph plus temporal encoder
   only if scenario scale and evaluation show that interactions need it.
5. **Use the estimator only to guide search.** Enable candidate pruning/ranking only if it
   improves held-out solution quality or reduces OSRD validation calls at the same
   quality. Revalidate each returned whole-timetable alternative in OSRD. Evaluate by
   held-out infrastructure and, when available, an entire held-out network.
6. **Consider LoRA only for a suitable pretrained transformer.** Simulation episodes
   are training data, not a LoRA adapter by themselves. LoRA may later adapt a pretrained
   language/sequence model for structured request parsing or concise result summaries;
   the current small GNN has no pretrained backbone that makes LoRA useful. Keep any
   language model outside feasibility and impact decisions, require schema validation,
   and never let it apply timetable changes.

Promote each model independently with its dataset manifest, feature/action schema,
training code, held-out report, calibration/coverage results, and rollback artifact. More
data or a larger model is not a promotion criterion without measured gains on the
intended task.

## 5. Training-data pipeline

### Sources

1. **Synthetic OSRD simulation data (starting point):** generated from RailJSON infrastructure and rolling stock, timetable schedules, pathfinding results, standalone and multi-train simulations, conflict outputs, and parameterized perturbations (late departure, dwell variance, closure, speed restriction, degraded capacity). OSRD’s realistic representative rolling stocks and test infrastructures enable reproducible initial datasets, but toy/test data is not evidence of real-world generalization.
2. **Historical operational data (needed for real operations):** planned and actual train event times, cancellations, train identity/class, rolling-stock assignment where permitted, resource occupations, incidents/temporary restrictions, and service priority/connection impact. Each operator feed needs a source adapter, legal basis, data-sharing agreement, data-quality profile, timestamp/timezone normalization, and retention policy.
3. **Human decision data:** candidate options shown, accepted/rejected edits and reason codes, overrides, and realized outcome. Rejections are not automatically “bad labels”; they may reflect constraints not present in model features. Treat as preference evidence with provenance.

### Public European data reviewed (2026-10-13)

**Recommended first external source: Belgium's RIDE release and its Infrabel inputs.** The RIDE benchmark reports three years (2023–2025) of passenger operations, with 94.5 million train-stop events, 3.6 million journeys, infrastructure tables, inferred network topology, and weather. Its Silver release is a reusable relational layer; Gold Lite is a smaller standardized delay-prediction benchmark with a temporal split (training snapshots from 2023–2024 and test snapshots from 2025), predicting delay at the next 15 scheduled events (about 40 minutes on average). The public RIDE datasets are CC BY 4.0; preserve attribution to RIDE, Infrabel, and Open-Meteo. The upstream Infrabel raw punctuality, operational-point, and line-section sources are CC0. See the [RIDE paper](https://arxiv.org/abs/2606.05070), [RIDE code and release links](https://github.com/orailix/ride), [Silver data](https://huggingface.co/datasets/orailix/ride-silver), and Infrabel's [monthly raw punctuality files](https://opendata.infrabel.be/explore/dataset/stiptheid-gegevens-maandelijksebestanden/).

Use the compact shared Gold Lite evaluation table first to inspect the actual schema and establish an external event-level delay forecast benchmark. The table has 623,314 rows, a `last_known_delay` in seconds, and `future_delay_1` through `future_delay_15` targets in seconds, paired with future event timestamps, operation-node IDs, and event types. Its 2025 evaluation period is temporally held out from the 2023–2024 training snapshots. Begin with a last-known-delay persistence reference and report MAE, RMSE, signed bias, and valid counts by horizon; do not train against this evaluation table. On the pinned table, this baseline scored 97.5 s MAE and 236.9 s RMSE across 6.53 million valid horizon targets; MAE rose from 56.0 s at event 1 to 133.2 s at event 15. Use this fixed test result for final comparison, not model or hyperparameter selection. The release's full tabular arrays total multiple gigabytes, so fetch only the pinned core table initially; expand ingestion only after confirming required fields and a benchmark protocol. Then use Silver or the CC0 Infrabel sources to build an OSRD-specific normalized event and infrastructure layer. Infrabel's monthly files include planned and observed arrival/departure times, delays in seconds, train number, operator, measuring-point identifiers and line identifiers; the portal advertises a long raw history, but ingestion should pin and verify each monthly file rather than assume uninterrupted coverage. The infrastructure data provide operational points and line sections, not a ready-made OSRD RailJSON or a complete signal/block-resource model. RIDE's inferred paths are valuable features but should retain their provenance and confidence.

These real records improve observed-delay labels and calibrate run/dwell distributions. The local Infrabel pipeline now has all twelve months of 2024 normalized with source hashes and clock-agreement diagnostics; its annual profile has 1.2 million origin delays and 20.8 million planned dwell observations. The current 160 OSRD episodes were conditioned on January 2024 only, so a fresh OSRD collection and fit are still required to consume the annual profile. These records do **not** say what would have happened under a changed timetable, and observed punctuality alone cannot label OSRD conflicts or prove a recovery plan caused an improvement. Keep source-event forecasting labels separate from paired OSRD counterfactual labels, and use OSRD validation for candidate-plan feasibility. RIDE is passenger-focused; freight, cancellations and planning constraints need separate coverage.

**Switzerland is a strong secondary generalization source, with more caveats.** [opentransportdata.swiss GTFS static](https://opentransportdata.swiss/en/cookbook/timetable-cookbook/gtfs/) covers the public-transport timetable year and provides stop, route and trip schedules; it is useful for timetable-pattern and cross-country transfer features, not detailed infrastructure feasibility. Its [daily actual-data v2](https://data.opentransportdata.swiss/en/dataset/ist-daten-v2) has train calls, planned times, forecast/observed times, cancellation/additional-run flags, and event status. The provider warns that “actual” can be the last forecast or an estimate, not a measured event, and calls out missing real-time coverage and quality edge cases. The catalog is daily updated and surfaces recent-day files; I have not verified a stable multi-year historical archive, so do not treat it as one without checking archive coverage. Switzerland also publishes [annual train counts by section](https://opentransportdata.swiss/en/cookbook/historic-and-statistics-cookbook/anzahl-zuege/), including passenger/freight counts and gross tonnage, but coverage excludes some infrastructure managers. File-based data are freely usable under the platform terms; cite the source. Service APIs require registration/access codes and are rate limited.

The Belgian SNCB timetable/real-time feed is another possible source, but it is provided under a separate [standard data license](https://www.belgiantrain.be/-/media/files/pdf/support/public-data/gtfs-open-data-standard-license.ashx?la=en&v=3ba4f2737ca34e82aec7063a4df269de), not the CC0 Infrabel terms; review that license before ingesting or redistributing it.

**Data roadmap:** (1) fetch and pin the RIDE Gold Lite core evaluation table and inspect its schema; (2) implement an external delay benchmark only for supported fields; (3) inspect Silver and document field mappings and quality flags before adding a larger importer; (4) map Belgian station/line identifiers to an OSRD infrastructure snapshot only for corridors with defensible topology coverage; (5) add Swiss data as a separate source/domain holdout, not as a silent merge; and (6) retain synthetic OSRD paired replays for conflict and timetable-change counterfactuals. Do not commit bulk third-party data to this repository; store checksummed manifests and reproducible download/transform code instead.

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
| Timetable-change impact | For the same pinned baseline, per-service/event delay and conflict deltas after adding a service or applying a candidate edit; include the exact action, affected set, objective values, and OSRD validation outcome | Paired OSRD baseline/candidate simulation; later compare with realized operator outcomes where identifiable |
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
| 1. Dataset and simulator replay | 4–6 weeks | Versioned RailJSON/timetable snapshot pipeline; seeded service insertion, retiming, dwell, and route-choice perturbations; paired baseline/candidate OSRD outputs and labels; leakage-safe splits; reproducible conflict/impact report |
| 2. Forecast baseline | 4–6 weeks | Delay quantile baseline (GBDT/TCN), calibration, corridor/time holdouts, applicability gate; no UI action path |
| 3. Service-planning API and deterministic search | 6–8 weeks | Typed line/stops/window/frequency request; generate and pathfind new-service schedules; bounded edits to eligible existing services; full-timetable OSRD validation; immutable job record; UI shows explicit deltas and unmet constraints |
| 4. Action-conditioned impact model + planner UI | 6–8 weeks | Train on paired baseline/change episodes; improve search quality or reduce OSRD validation calls against a no-ML baseline on held-out scenarios; show network-wide effects and uncertainty; explicit draft apply/undo; authz/audit tests |
| 5. Shadow pilot | 8–12 weeks | Operator feed agreement and source adapter; live shadow forecasts, drift/latency dashboards, incident review; no operational actuation |
| 6. Controlled decision-support pilot | 3–6 months | Operator-specific assurance case/procedures; planners trial service requests and network-change alternatives; measured improvement, acceptable overrides and no invalid surfaced plans; staged approval |

Timelines assume a small cross-functional team (rail operations/domain expert, OSRD Rust engineer, ML/data engineer, frontend engineer, safety/security review). They are planning estimates, not commitments; access to real operational data is the largest schedule dependency.

**Earlier v3 model status (historical):** the repository includes an OSRD path-resource-aware
GNN trained for one narrow task: ranking services likely to become newly conflict-
involved after inserting a proposed service into a mixed-route timetable. Version 3 was trained
on 240 episodes across two medium corridors and one Y branch, validated on 160
episodes across medium and Y, and tested on 240 episodes across a held-out medium
corridor, `small_infra`, and the opposite Y branch. Held-out conflict AP was 0.688
on the medium corridor (linear 0.621), 0.392 on `small_infra` (linear 0.253), and
0.874 on the opposite Y branch (linear 0.770). A positive-slope Platt calibrator,
fit on validation groups only, reduced held-out Brier from 0.108 to 0.054; the
small fixture remains worse than linear while the Y fixture improves. Thresholds
are available on medium and Y validation infrastructures but suppressed on
`small_infra`. Arrival-delta MAE (0.81 s) lost to zero-change (0 s), so that head
is disabled. The artifact is eligible only for offline conflict-score ranking.
All fixtures remain synthetic; the live planner continues deterministic objective
ordering and OSRD validation. Operator data, reliable cross-network calibration,
useful arrival estimates, and held-out search-benefit evaluation remain before any
learned score is enabled.

**Historical v5 model status:** the expanded experiment used 1,120 paired episodes:
560 train, 160 validation, and 400 test across six OSRD fixtures. It adds four
ETCS training cohorts and held-out overlapping-routes and nested-switch tests.
Aggregate held-out conflict AP is 0.767 versus 0.641 for linear; calibrated Brier
is 0.041 versus 0.045. The offline promotion gate requires at least a 0.05 AP gain
over references on every test group. It fails: nested-switch and opposite-Y gains
are 0.022 and 0.032, so the v5 predictor suppresses threshold decisions. Arrival
predictions remain disabled. All data are synthetic; real/operator networks and
measured end-to-end search benefit remain outstanding.

**Historical action-impact model status (v11):** the graph now uses explicit shared
track/route/block counts and candidate-service interaction edge features. The latest
full-corpus artifact uses 1,145 training episodes, 20 validation episodes, and 120
held-out episodes in three synthetic scenario groups. Aggregate conflict AP is 0.963
versus 0.924 for logistic regression; Brier score is 0.102 versus 0.141. Arrival-delta
MAE is 13.58 s versus 80.26 s for zero-change. AP gains are +0.070 on sparse medium
traffic, +0.027 on nested-switches, and +0.010 on the tiny fixture. A second-seed refit
on the same split selects epoch 9 and scores aggregate AP 0.966 versus 0.924 logistic,
with per-group gains of +0.067, +0.014, and +0.024. The required +0.05 per-group margin
still fails, so learned ranking remains disabled; numeric arrival predictions
are scoped to the validation infrastructure. The artifact does not construct a timetable
or choose a route, and every candidate still requires whole-timetable OSRD validation.
All action-impact labels and fixtures remain OSRD simulations. Full-year Infrabel actuals
condition some Belgian replay delays and dwell; they do not turn OSRD counterfactuals
into measured outcomes. A one-day Swiss Actual profile now conditions v25 Swiss replay
inputs, but actual Swiss events have not been mapped to RailJSON services or used as
counterfactual impact labels. Swiss open-data assets can support further network-level
evaluation:
[swissTNE Base](https://www.swisstopo.admin.ch/en/landscape-model-swisstne-base) supplies
interconnected transport geometry, while the [Swiss train-count dataset](https://opentransportdata.swiss/en/cookbook/historic-and-statistics-cookbook/anzahl-zuege/)
provides annual directional section totals for a subset of infrastructure managers.
Neither supplies the detailed track, route, signaling, timetable, or paired intervention
outcomes required to simulate OSRD counterfactuals; they cannot replace a licensed,
validated RailJSON network and OSRD-generated labels.

**Latest action-impact status (v12 route-topology expansion, 2026-10-16):** feature-v8
GNN adds 160 tiny-topology and 80 alternate-platform-route training episodes to the
previous 66 cohorts. The original validation group and all 20 test groups remain fixed:
1,565 train / 20 validation / 780 test episodes across 43 / 1 / 20 groups. Aggregate
heldout conflict AP is 0.987 versus 0.948 for logistic regression and Brier is 0.061
versus 0.121. The minimum per-group AP gain is +0.003, still below the +0.05 release
gate. On 40 matched candidates each at 120-second and 600-second frequency, phase-choice
accuracy is 56.25% and 51.25%, respectively (half credit for ties). Mean new-conflicted
services remain close to random choice, and positive arrival delay is unchanged between
paired phases. Logistic AP is already 0.9915 on both 120-second platform test groups,
which caps possible improvement below the +0.05 gate even with perfect predictions.
Candidate ranking remains disabled pending independently reviewed release criteria and
real-network benefit evidence. The latest artifact and reports are in
`ops-intelligence/models/trained/service-plan-v12-route-topology-expansion/`; new training
episodes are under `ops-intelligence/datasets/replay-service-plan-v5/`.

**v13 Infrabel-2025 expansion (2026-10-17):** downloaded, normalized, and checksummed
the twelve 2025 Infrabel monthly passenger files (22,427,357 events), built a separate
origin-delay profile (1,178,119 samples), and collected 160 paired service-plan training
episodes across four OSRD fixtures. The fit retains the v12 validation and test groups
exactly (1,725 train / 20 validation / 780 test across 47 / 1 / 20 groups). Aggregate
test AP is 0.986 versus 0.953 logistic and Brier is 0.069 versus 0.116; arrival MAE is
3.67 s versus 46.45 s for zero-change. Minimum per-group AP gain is +0.0018, so the
unchanged +0.05 gate still fails. These Belgian observations condition synthetic OSRD
baseline inputs; they do not provide Belgian network topology or counterfactual labels.
Artifact: `ops-intelligence/models/trained/service-plan-v13-infrabel-2025-expanded/`.

**v14 OSM-derived smoke cohort (2026-10-17):** added 40 training-only episodes from
one imported OSM-derived Belgian track segment, conditioned with the separate 2025
Infrabel profile. Training retained the exact v13 validation and test groups (1,765 train
/ 20 validation / 780 test). Test conflict AP is 0.9875 versus 0.9528 logistic; Brier is
0.0668 versus 0.1158; arrival-delta MAE is 4.36 s versus 46.45 s zero-change. The
minimum group AP gain is +0.0024, so candidate ranking is still gated off. Phase-choice
accuracy on the existing unrelated synthetic holdouts is 56.25% at 120 s and 48.75% at
600 s; mean conflicts are effectively at random. This is not a held-out Belgian-network
evaluation: cross-track pathfinding on the large inferred OSM infrastructure is still
unstable, and the only OSM cohort is a short same-track case. Artifact:
`ops-intelligence/models/trained/service-plan-v14-osm-brussels-midi/`.

**v15 route-conditioned training and holdout (2026-10-17):** added matched TC0/TC1
platform route actions on the medium synthetic fixture using the 2025 Infrabel profile.
The second seed contributed 80 training episodes; a separate 40-pair first-seed cohort
was held out by scenario group, with exact baseline and perturbation matching verified.
V15 uses 1,845 train / 20 validation / 860 test episodes across 50 / 1 / 22 groups.
Test conflict AP is 0.9844 versus 0.9527 logistic, Brier 0.0670 versus 0.1153, and the
minimum group AP gain is +0.00094, so candidate ranking remains gated off. Both held-out
routes produce identical per-train conflict and arrival labels; route-choice accuracy is
50%, with no measured benefit. This tests the paired route-action pipeline on one synthetic
infrastructure only. Artifact: `ops-intelligence/models/trained/service-plan-v15-route-choice/`.

**v16 OSM adjacent-track holdout (2026-10-17):** collected 40 paired service-plan
episodes on a short OSM-derived path crossing from track `1122923669-0` to
`116000117-0`. The OSRD k-shortest endpoint, block pathfinding, and full simulations
completed on this segment; the longer Brussels corridor still times out. The group is
test-only, preserving v15 train/validation and prior test groups (1,845 / 20 / 900 episodes
across 50 / 1 / 23 groups). On this OSM group, GNN and logistic conflict AP both equal
1.000; Brier is 0.197 versus 0.561. Arrival MAE is 72.86 s versus 20.44 s zero-change.
The minimum test-group AP gain is zero and the arrival head fails to transfer. This is a
short same-import track holdout, not an independent network test. Artifact:
`ops-intelligence/models/trained/service-plan-v16-osm-adjacent-holdout/`.

These experiments still use synthetic OSRD simulator labels. The OSM-derived segment and
medium-fixture route alternatives add topology-shaped inputs, but no route choice has
shown benefit; the v16 OSM holdout does not beat the logistic reference on AP, and no
independent operator network has been held out. The long-term design
still needs independently licensed networks, broader route/stop/resource alternatives,
robust objective gains, calibrated uncertainty, and measured end-to-end search benefit.
The model remains a surrogate; OSRD must simulate and validate each whole draft.

**Geographic OSM follow-up (2026-10-17):** v17 added 480 training episodes from 12
southern switch locations and 160 test episodes from four northern locations. V20 then
added 480 matched phase-180 training episodes. V21 added 960 episodes at 60s and 120s on
those same training switches while preserving the prior test groups. V22 added a frozen
fresh test of 160 matched 60s-versus-120s pairs at four previously unused northern
switches. The final v22 split has 3,765 train / 20 validation / 2,340 test episodes across
98 / 1 / 55 groups. Overall test conflict AP is 0.969 versus 0.882 for logistic; Brier is
0.079 versus 0.134. On the fresh phase holdout, v22 reached 66.56% pairwise objective
accuracy and averaged 2.069 new conflicts versus 2.288 for random. Three locations
improved and one tied.

V23 added 960 matched 120s/600s frequency training episodes and a fresh 160-pair
frequency-choice holdout at four more northern switches. V23 has 4,725 train / 20
validation / 2,660 test episodes across 122 / 1 / 63 groups. It reached 90% frequency-
choice objective accuracy and averaged 2.675 new conflicts versus 3.241 for random; all
four locations improved and positive arrival delay was unchanged. Overall test conflict
AP is 0.965 versus 0.892 for logistic. The minimum held-out-group AP gain is -0.0154, so
ranking and pruning stay disabled. These are simulated outcomes on one OSM-derived
Belgian infrastructure. Broader operational regimes, independent networks, and end-to-
end planner search benefit remain necessary. Reports are in
`ops-intelligence/models/trained/service-plan-v23-frequency-generalization/`.

**Independent-network frequency holdout (v24, 2026-10-17):** a dated Switzerland OSM
extract was converted to RailJSON and imported as infra 17. Four geographically separated
Swiss switch groups contributed 160 matched 120s/600s frequency pairs (320 episodes), all
held out; training remains Belgian. V24 achieves 80% pairwise objective accuracy and
averages 1.831 new conflicts versus 2.238 for random, with equal positive arrival delay.
Pooled held-out conflict AP is 0.961 versus 0.890 for logistic, but the minimum held-out
group AP gain is -0.0176 and the infrastructure is outside the model's calibration scope.
Ranking remains disabled. These synthetic OSRD labels test topology transfer only: Swiss
actual delay data was not acquired, and the replay uses unconditioned disturbances.
Artifacts and ODbL source checksums are in
`ops-intelligence/models/trained/service-plan-v24-swiss-network-holdout/` and
`ops-intelligence/data/rail/osm-switzerland/source-manifest.json`.

**Swiss profile-conditioned check (v25, 2026-10-17):** one day of Swiss Actual data v2
(2026-10-14) was processed into an empirical profile that samples only status `REAL` at
the first scheduled Swiss-stop departure per service; forecast and estimated statuses are
excluded. It contains 146,533 departure-delay samples and 2,250,129 scheduled dwell
samples. A new 160-pair Swiss frequency-choice holdout applies this profile only to OSRD
simulation inputs; no Swiss episodes enter training. V25 reaches 92.5% pairwise objective
accuracy and averages 2.950 new conflicts versus 3.634 for random, with equal positive
arrival delay. Test conflict AP is 0.962 versus 0.895 for logistic; the minimum held-out
group AP gain remains -0.0176, so planner ranking stays disabled. The one-day profile and
four locations do not establish seasonal or operational transfer. The v2 publisher
reported malformed 2026-10-12 rows; that date was excluded. Profile source and filtering
provenance are in `ops-intelligence/data/rail/swiss-actual-v2/source-manifest.json`.

**First Swiss-trained multi-network fit (v26, 2026-10-17):** 960 paired-frequency
episodes were generated at 12 geographically separated southern Swiss switch locations,
conditioned on the Swiss profile. One location (both frequency arms, 80 episodes) is
reserved for validation; the other 11 locations (880 episodes) join the Belgian training
corpus. All 71 prior held-out scenario groups remain frozen, including the northern Swiss
topology and profile-conditioned frequency tests. The training run is checkpointed and
completed after 55 epochs (best epoch 30). The full test set scores conflict AP 0.978
versus 0.903 for logistic, Brier 0.0662 versus 0.1278, and arrival MAE 3.27 s versus
38.48 s for zero-change. The weakest per-group AP difference is -0.496 on the Midi–Nord
adjacent-track group, so the promotion gate still fails. On the 160 profile-conditioned
northern Swiss frequency pairs, v26 matches v25 at 92.5% pairwise accuracy and averages
2.950 conflicts versus 3.634 for random. That benchmark has only four locations and the
observed outcome is still an OSRD simulation; do not enable planner ranking. The model
artifact and Swiss choice summary are under
`ops-intelligence/models/trained/service-plan-v26-swiss-multinetwork/`.

**Multi-season Swiss profile fit (v27, 2026-10-17):** the official August 2025 and March
2026 monthly archives add 62 service days and 7,708,831 REAL first-departure delay
observations to the profile source population. A bounded 250,000-delay / 250,000-dwell
sample conditions a second matched seed of 960 Swiss replay episodes; v27 trains on 6,485
episodes, validates on 180, and tests on the same 3,300 frozen episodes. Test conflict AP
is 0.978 versus 0.907 for logistic, Brier is 0.0686 versus 0.1266, and arrival MAE is
2.66 s versus 38.48 s for zero-change. The minimum test-group AP difference is -0.0048,
so the +0.05 per-group ranking gate still fails. On the one-day-profile Swiss test v27
again chooses the lower-conflict frequency in 92.5% of 160 pairs, averaging 2.950
conflicts versus 3.634 for random. More historical operational marginals stabilize group
performance but did not improve this choice benchmark. The data source and daily member
hashes are in `ops-intelligence/data/rail/swiss-actual-v2/source-manifest.json`; raw ZIPs
remain local and ignored by Git.

**Modal-data correction (2026-10-17):** a later schema audit found Swiss Actual v2
contains multiple transport products. The v25-v27 adapter did not filter `PRODUKT_ID`,
so these Swiss-conditioned replay contexts mixed rail and non-rail delays. Their reported
Swiss results remain historical but are not rail-only benchmarks. The rebuilt 2026-10-14
profile filters `Zug`; it retains 14,026 `REAL` first-departure samples and excludes
2,423,561 non-rail rows. The old artifacts are retained for audit and their profile-
conditioned cohorts are excluded from the v28 fit.

**Rail-only multi-frequency GNN (v28, 2026-10-17):** the Swiss adapter streamed all 15
monthly Actual v2 archives from 2025-07-13 through 2026-09-30 (445 daily CSV members,
1,054,186,238 rows). It retained only `PRODUKT_ID=Zug`, excluding 974,785,755 non-rail
rows; the empirical profile uses a bounded 250,000-delay / 250,000-dwell sample from
6,117,227 real first departures and 57,518,355 dwell observations. These observations
condition OSRD simulations; they are not mapped to OSRD train identities or counterfactual
impact labels. Eight southern switch sites generated 1,680 training and 240 validation
episodes; four northern sites generated 960 held-out episodes across 120/180/240/300/420/
600-second service frequencies. The held-out episodes use the separate 2026-10-14 rail-only
profile. The complete split contains 6,405 train / 260 validation / 3,620 test episodes.

V28 selected epoch 35 of 60. On all held-out episodes, conflict F1 is 0.900 and average
precision is 0.978 versus 0.929 for logistic regression; arrival-delta MAE is 2.55 s
versus 38.75 s for a zero-change reference. In the Swiss frequency-grid study, the model
matched the best OSRD-simulated lexicographic conflict/delay outcome for 150 of 160
contexts (93.75%), with 75.4% pairwise accuracy over 2,400 comparisons. It selected
2.863 conflicts on average versus 3.500 for a fixed 300-second choice, but its 2.863
conflicts and 93.75% exact-match rate tie a fixed 600-second choice. The candidates all
had identical realized positive arrival-delay sums, so this grid mainly tests conflict
ranking. It does not encode passenger demand or protect a requested service intensity.
The per-group AP margin gate remains false (minimum -0.0089), and the service API keeps
candidate ranking disabled. V28 is a stronger research artifact, not a promoted planner
decision model. Its checkpoint, split report, and detailed grid evaluation are under
`ops-intelligence/models/trained/service-plan-v28-swiss-railonly-fullmultifrequency/`;
archive and daily-member hashes are recorded in
`ops-intelligence/data/rail/swiss-actual-v2/source-manifest.json`.

**OSM corridor crop follow-up (2026-10-17):** a bounds-based RailJSON crop retained 711
tracks and 515 complete switches around Midi–Nord, discarded generated routes for OSRD to
rebuild, and imported as local infra 14. Shortest-path search from Midi to Nord exceeded
30 seconds and Editoast stopped responding, so this crop produced no route or simulation
labels and is excluded from training. The crop utility and checksum are recorded alongside
the local OSM source snapshot. Treat this as a route-search scalability blocker; do not
represent the corridor as validated network-transfer data.

### Service-planning vertical-slice gates

Deliver the long-term capability in increasing scope so each step is useful and
measurable:

1. **Schedule a service on an otherwise empty test timetable.** Resolve the requested
   stops and time window, expand a frequency into departures, request supported paths,
   and return the OSRD-simulated draft. Report unsupported or infeasible requirements.
2. **Fit it into an existing timetable without changing other services.** Show resource
   conflicts and the reason no-change insertion does or does not work. This becomes the
   baseline against which improvement is measured.
3. **Propose bounded changes to existing services.** Start with departure shifts, then
   supported dwell and path alternatives. Respect protected-service and per-schedule
   change caps. Return all new and modified schedules together as a versioned draft.
4. **Rank alternatives by network impact.** Train the action-conditioned estimator on
   paired OSRD runs; compare against persistence/GBDT impact baselines and deterministic
   search. Require improvement in search efficiency or solution quality on held-out
   scenarios before enabling model-based pruning.
5. **Add planner review and controlled application.** Compare alternatives, changed
   services, validation evidence, uncertainty, and objective trade-offs. Applying a
   selected result creates a reviewable draft with explicit undo and stale-snapshot
   rejection.

At every gate, a plan is not called feasible until OSRD has validated the entire
candidate timetable. If the request cannot be met within its declared bounds, return the
best validated partial alternatives (if policy allows) and a structured explanation of
the unmet frequency/window/route constraints; never invent changes outside the allowed
service set.

## 11. Next implementation slice: action-conditioned service planning and data

The repository now has an initial bounded service-planning loop. It expands an explicit
OSRD path, dwell pattern, UTC window, and frequency into calendar departures; searches
bounded first-departure phases; generates insertion-only, single-train, pairwise, and
coordinated departure-shift candidates for explicitly eligible trains; and validates each
complete temporary timetable through OSRD. The source timetable remains unchanged. This is
deterministic bounded search, not a learned generator or globally optimal solver. Complete
plans can compare planner-marked optional dwell stops jointly with route alternatives and
departure phases. The system does not invent new stop locations, and the current GNN remains
advisory-only and gated off.

**Coupled-plan model status (v34, 17 October 2026):** the paired training set contains
31,920 checksum-verified OSRD episodes: 15,120 training, 6,720 validation, and 10,080
test episodes. Spatial splits cover six southern Swiss training sites, two southern
Swiss validation sites, four newly collected Swiss test sites, six southern Belgian
training sites, four Belgian validation sites, and four newly collected Belgian test
sites. Each matched
request has seven alternatives (no retiming, one-service +900s, and all-service shifts
of +300/+450/+600/+900/-450s) under 300s or 600s requested frequency and three departure
phases. Belgian replays use Infrabel's 2025 origin-delay profile; Swiss replays use the
rail-only 2025-07–2026-09 profile for training/validation and a separate 2026-10-14
profile for test. These profiles condition synthetic OSRD scenarios; they do not label
real counterfactual conflicts.

The two-layer graph attention model reaches test conflict AP 0.899 versus 0.813 for its
logistic reference and arrival-change MAE 7.60s versus 401.79s for zero-change. Its
minimum per-group AP gain is -0.214, below the +0.05 promotion rule, so online ranking
remains unavailable. On matched-plan ordering, action-cost weight 0.1 selected using
validation only gives 41.0% exact objective matches on 960 new Swiss contexts and 84.0%
on 480 new Belgian contexts. It orders the first OSRD-accepted candidate in 1.01 and 2.00
simulations on average, respectively, versus 4.34 and 6.73 for minimum-edit ordering.
Eighty Belgian contexts have no zero-conflict alternative in the tested action grid.
These call-count comparisons are retrospective because the candidates were all simulated
to create labels. OSRD simulation remains the feasibility authority; the learned
top-ranked plan is not guaranteed conflict-free.

This completes a fresh-holdout, multi-network, simulation-trained impact-ranking
research artifact, not the full operational planning model. Remaining production and
evidence work is:

1. Obtain operator-authorized historical timetable snapshots, actual movement outcomes,
   and planner choices so labels include observed outcomes and trade-offs rather than only
   counterfactual OSRD simulations.
2. Add station-mapped stop patterns, route alternatives, service-day calendars, rolling
   stock constraints, and longer corridor interactions to the candidate data. Verify each
   generated route through OSRD pathfinding and simulation.
3. Measure model-guided search prospectively on untouched requests: wall-clock and OSRD
   simulation calls, feasible-plan yield, objective quality, and planner acceptance.
   Calibrate uncertainty and preserve the existing per-group promotion threshold.
4. Connect a model to planner candidate ordering only after those gates pass. Until then,
   keep deterministic bounded search and OSRD validation in the planning API, and use the
   GNN artifacts for offline research.

The service-plan API now queries OSRD's Yen k-shortest-path endpoint (maximum five paths),
converts route track ranges into ordered track-offset waypoints, and validates complete
candidates through OSRD. It currently supports track-offset waypoints only; operational
point parts are not yet resolved into track locations. OSRD's topological path ranking does
not consider rolling-stock constraints, so final candidate pathfinding and simulation
remain mandatory. No multi-route corridor smoke run has yet proved these constraints
faithfully force each returned route, and the training cohorts do not yet cover route
alternatives. Treat this as an implemented but unevaluated candidate generator, not a
learned or operational route planner.

**OSRD coordinate contract correction (2026-10-17):** timetable and `/pathfinding/blocks`
track offsets are millimetres; the separate `/infra/{id}/pathfinding/?number=` Yen endpoint
accepts and returns metres. The client now divides request offsets by 1,000 and multiplies
returned range midpoints by 1,000 before constructing timetable waypoints. A corrected
Belgium route-waypoint smoke replay passed OSRD pathfinding and simulation. This validates
one route, not distinct-route selection; route-choice cohort collection and a forced-route
multi-alternative evaluation remain outstanding.

The frequency-action data is reproducible from `datasets/replay-insertion-v5/`. It adds
108 paired OSRD episodes for repeated departures and an intermediate-stop cohort, with
v6 action features for cadence, window, count, dwell, and service slot. Model v8 trained
on 1,228 episodes has aggregate held-out AP 0.732 versus 0.697 for logistic regression,
but misses the 0.05 per-group promotion margin (minimum 0.007 on nested switches). Its
arrival head also loses to zero-change. Keep it gated off; increase frequency and
route-choice coverage and collect additional held-out operator or infrastructure data.

This ordering tests the core hypothesis—whether OSRD plus available operations data can support useful, generalizable predictions—before expanding the prototype into a production service boundary. The current artifacts do not satisfy these completion gates: the three v11 test groups are scenario holdouts on synthetic fixtures, not independent operator networks; only service insertion, stop, frequency, and retiming variants are labeled; and end-to-end search benefit has not been measured.

The first planning-core increment is a deterministic departure-shift proposal generator for incident-affected schedules. It excludes protected trains, caps each shift and candidate count, and binds candidate IDs to the full infrastructure, timetable, and OSRD version snapshot. These proposals are deltas only: the service must apply each to an isolated full timetable and use OSRD's simulation and conflict outputs before any plan can be returned.

The follow-on integration slice reads timetable-to-schedule-set relationships from the pinned Editoast route, copies all regular and paced schedules into temporary unpublished records, recreates paced exceptions against candidate schedule IDs, validates base and exception simulation summaries plus conflict detection, and cleans up the candidate records. Explicit exception departure times shift with their parent paced schedule. A local OSRD run validated a paced candidate with one modified occurrence and zero conflicts; the temporary candidate records were removed and the source revision remained unchanged.

Candidate selection now supports explicit operator objectives: minimize total shift, minimize the number of changed services, or minimize priority-weighted shift. Weighted ranking requires a positive priority for each affected unprotected service; protected services remain excluded from candidate generation. The API reports the selected objective, per-candidate score, total proposal count, and validation count so operator policy and validation effort are auditable.

The planner also accepts hard per-service retiming caps through `max_shift_seconds_by_schedule` (0–7200 seconds). These caps are applied while constructing proposals, before objective ranking and OSRD validation; omitted schedules inherit the request-wide `max_departure_shift_seconds`. A zero cap makes a service ineligible for edits without changing its protection status or requiring an objective weight.

Phase 3 now has a validated pass-through baseline: the exact source timetable is copied to an isolated unpublished timetable and receives the same whole-timetable simulation and conflict checks as edited candidates. The completed response and original typed request are stored in an append-only SQLite journal with input/result SHA-256 digests, keyed by request ID; retries with the same ID and payload return the stored result, while a changed payload conflicts. A GET route retrieves the record. This meets the prototype planning-slice audit requirement; production must move authoritative records and authenticated operator identity into Editoast persistence and its authorization boundary.

Phase 4 starts with an offline extraction path from these immutable records into scenario-grouped candidate-feasibility examples. Labels are exactly OSRD validation outcomes; the feature schema does not claim operator utility or acceptance. Since only candidates selected for validation are recorded, the first learned score can prioritize validation only and cannot bypass OSRD or be described as an unbiased plan-quality ranker. Broader candidate exploration and outcome/preference data are prerequisites for a utility ranker.

The first feasibility model is a regularized logistic classifier over that fixed candidate feature schema. Its offline report compares scenario-macro top-1/top-3/top-5 validation yield against the current objective-score order and reports an oracle bound. Promotion and online planner integration are deliberately separate; this model is not loaded by the service, and it cannot certify candidate feasibility.

To reduce top-objective selection bias in future feasibility examples, planning policy can choose `candidate_validation_order=diverse`. The deterministic round-robin covers different edited-service sets before spending additional validation calls on variants of the same edit set. It does not change how returned alternatives are objective-ranked and does not relax OSRD acceptance checks.

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

## 13. Direct plan-level graph scorer status

The v34 service-impact network predicts effects per train and then relies on a
validation-selected action-cost blend for coupled candidate ordering. The new
[`plan-gnn-v1 model card`](../../ops-intelligence/models/trained/plan-gnn-v1/MODEL_CARD.md)
records a separate action-conditioned plan scorer: two edge-aware GATv2 layers,
mean/max graph readout, plan-level conflict and positive-delay heads, and a direct
pairwise score trained across alternatives sharing a scenario, seed, and
episode index. Its explicit supervision is lexicographic—minimize newly
conflicted services, then positive existing-service arrival delay for
equal-conflict alternatives. At inference, the lower direct score orders only
the candidates supplied to it; it is not a probability or timetable generator.

The fit uses the full available coupled corpus: 15,120 training, 6,720
validation, and 10,080 test candidate episodes, or 2,160 / 960 / 1,440 matched
contexts across Swiss and Belgian OSRD simulations. The test partition has
already been inspected in prior experiments, so its v1 ranking metrics are
diagnostic only. Validation pairwise concordance is 0.859, and top-ranked plans
have the minimum simulated conflict count in 92.8% of contexts, compared with
2.6% for minimum-edit ordering. The model has substantial delay-head error and
has not been assessed prospectively with planners.

Action coverage is deliberately narrower than the target planning task: only
300/600-second frequencies, 0/60/120-second departure phases, and seven
retiming alternatives are represented. All examples use no added stops, zero
dwell, and no route alternative. Consequently the fit does not establish that
the model can infer optional stops, choose among routes, or assemble a complete
timetable from a line-level request. The deterministic planner can search only
optional intermediate stops explicitly marked by the planner and still requires
OSRD to validate every candidate.

This advances the synthetic action-ranking model but does not satisfy the
operational data requirement. The public Swiss/Belgian actuals only condition
simulations; the labels for alternative actions still come from OSRD. Completing
the operational model requires operator-authorized timetable snapshots,
realized movements, planner options and decisions with reasons, broader route
and stop-pattern coverage, freight/cancellation cases, and a new untouched
spatial/network holdout. Keep the direct scorer out of the planning API until
those labels, the fresh evaluation, and the prospective search-benefit gate are
available. OSRD remains the only feasibility authority.

## 14. Joint multi-service calendar planning increment

`POST /v1/operations/complete-timetable/plans` accepts a bounded set of up to ten
calendar service patterns on one exact snapshot. It generates combinations of per-pattern
departure phases and up to five OSRD route alternatives per service, then validates all
requested departures together through OSRD, including pathfinding, simulation, and
conflicts against the source timetable. The request can also include an explicit retiming
policy for eligible existing services; those deltas are combined with each requested
service set and validated together. Results report the selected route for each pattern,
edits, and OSRD-simulated arrival/running-time effects where available. A service may
specify a hard maximum end-to-end duration; every departure must have a simulated time
within that bound, otherwise the candidate is rejected or marked unverified. Intermediate
waypoints marked optional are searched as bounded stop-pattern choices alongside routes
and phases. For flow-on explanation, the planner compares source-snapshot and candidate
simulation times for every existing service, lists services with an explicit retiming or a
changed running time, and reports partial coverage when a comparison cannot be made. The request cap is
50 departures and at most 32 OSRD candidate validations; the response reports phase,
route, and retiming combination counts and whether the validation budget sampled the
combined search space evenly. This jointly assesses several service patterns and route
options rather than planning them in isolation.

The planner can optionally rank these complete candidates by aggregated station-patronage
coverage after OSRD feasibility and conflict checks. This is a deterministic service-
availability proxy, not a learned passenger-benefit estimate. The planner may compare
subsets of user-marked optional stops but does not invent stop locations or service
frequencies.
The current search is bounded deterministic generation, not learned generation, and the
plan-level scorer remains offline and gated. A selected candidate still needs planner
confirmation and draft creation, with OSRD validation repeated before applying it.
