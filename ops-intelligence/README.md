# OSRD Operations Intelligence

## Project goal

Help railway planners create and revise OSRD timetables from service requirements such
as routes, stops, operating windows, and frequency. The planner should generate and
compare complete timetable options, explain their trade-offs and every proposed change
to existing services, and use OSRD pathfinding, simulation, and conflict detection to
validate each option. A planner reviews the evidence and chooses whether to create a
separate OSRD draft scenario. AI should help explore and prioritize options and learn
from planner feedback; it must not replace OSRD validation or apply a plan without
planner review.

## Current implementation

The service generates bounded route, departure-phase, and existing-service retiming
alternatives, validates complete candidates with OSRD pathfinding, simulation, and
conflict checks, then lets the planner create a separate draft scenario. The source
timetable is never edited by the planning API or draft workflow. Deterministic ordering
remains the default. An optional research model can suggest among feasible single-
departure, direct-path alternatives, without changing their order or OSRD validation.
Learned ranking for complete frequency and retiming plans remains disabled until a model
passes its held-out evaluation and promotion gates.

When OSRD rejects a candidate, the planner retains structured pathfinding details and
explains actionable causes such as an unavailable or unauthorized rolling stock, invalid
path points, or a route that cannot be connected. OSRD remains the authority for these
validation outcomes.

New OSRD conflicts also return the existing train schedule IDs involved. The planner can
add those services to the eligible set and rerun its bounded retiming search, preserving
protected services and shift caps; the planner still reviews the jointly validated result
before creating a draft.

For requests that only change existing services, the planner also supports a retiming-only
scope. It explores bounded advances and delays for explicitly eligible trains, validates
each full timetable through OSRD, and reports simulated effects for changed or otherwise
affected services. The request creates no new departures; selecting an option can copy
the timetable into a separate draft scenario after snapshot and conflict checks.

The service is deliberately isolated from OSRD core. Its offline replay collectors and
models support evaluation of future planning assistance; they do not certify candidate
feasibility or replace OSRD validation.

The planner also has an optional natural-language request interpreter. After the user
checks the explicit external-processing consent, it sends only the request text to the
configured AI provider and returns a strict-schema draft with unresolved station and
rolling-stock terms, assumptions, and clarification questions. The output never includes
OSRD identifiers, never starts timetable search, and never creates an OSRD draft. The
interpreter is disabled unless `OSRD_OPS_PLAN_INTENT_ENABLED=true`,
`OSRD_OPS_PLAN_INTENT_PROVIDER`, and `OSRD_OPS_PLAN_INTENT_MODEL` are configured. The
`openai` provider uses the Responses API structured-output schema and requires
`OPENAI_API_KEY`. The `ollama` provider uses its local chat API with the same JSON Schema;
configure `OSRD_OPS_OLLAMA_BASE_URL` for the Ollama server. For local Podman development,
the optional `docker-compose.local-ollama.yml` overlay runs the official Ollama container
on an internal-only network. The planner joins that network and its normal OSRD network;
Ollama cannot access OSRD or host-loopback services. The overlay mounts the host's model
store read-only (override `OSRD_OLLAMA_MODELS_DIR` if Ollama stores models elsewhere), so
the configured model must already exist locally. The overlay defaults to `qwen3:1.7b` for
lower-latency interactive intent drafts; set `OSRD_OPS_PLAN_INTENT_MODEL` to another
installed model, such as `qwen3:4b`, if you prefer a larger model:

```sh
DOCKER_SOCKET="$XDG_RUNTIME_DIR/podman/podman.sock" podman compose \
  -f docker-compose.yml -f docker/docker-compose.front.yml \
  -f ops-intelligence/docker-compose.yml -f docker/docker-compose.demo-auth.yml \
  -f ops-intelligence/docker-compose.local-ollama.yml \
  up -d --build ops-intelligence ollama
```

Local Ollama requests allow up to five minutes by default, which accommodates CPU-only
structured output for multi-service drafts. Set `OSRD_OPS_OLLAMA_TIMEOUT_SECONDS` on the
planner service to tune the timeout from 1 to 900 seconds for the host's inference speed.

The base compose stack keeps the AI interpreter disabled. With this overlay, only the
request text the planner user explicitly consents to send is sent to the local model.
OSRD identifiers must still be selected from the live infrastructure and every extracted
field must be reviewed in the planner form.

The interpreter can also extract a requested departure-time retiming for a named
existing service. The browser must match that name uniquely against the currently loaded
OSRD timetable before it can become an eligible edit; direction and explicit upper bounds
are carried into the retiming policy. Exact shift amounts trigger clarification rather
than being treated as caps. Weekday selections are accepted only when grounded in the
request or deterministically derived from one explicitly stated operating date; the
planner's optional all-days behavior is preserved when no weekdays are stated.

After reviewing an option, planners can download the versioned review record or
explicitly submit it to `/v1/operations/service-plan-reviews`. Submitted records are
stored immutably in the local plan-audit SQLite journal with a content hash. The record
captures a planner's stated choice for research; it does not establish operational
acceptance or a realized outcome. No review is stored unless the planner submits it.

## Layout

```text
src/osrd_ops_intelligence/
  api/         internal request/response contracts (pydantic)
  service/     FastAPI app and OSRD client boundary
  features/    versioned replay and RIDE feature transforms
  models/      linear, quantile-MLP, replay-impact, and plan-level GNN models
  optimizer/   bounded departure-shift and service-retiming candidate generation
  training/    collectors, validation splits, baselines, and resumable trainers
  registry/    model manifests, promotion metadata (placeholder)
data_contracts/  pinned upstream OpenAPI artifacts
tests/           unit and contract tests
```

Repo-level companions: `../ops-data/` (generators, manifests; no licensed data),
`../tests/ai_operations/` (end-to-end OSRD fixture scenarios) and
`../docs/{architecture,safety-case,runbooks}`.

## Local development

Requires Python 3.11+ and `uv`.

```sh
uv sync --extra dev
uv run uvicorn osrd_ops_intelligence.service.app:app --app-dir src --reload --port 8095
```

OpenAPI is available at `http://localhost:8095/docs`; process health is at
`http://localhost:8095/healthz`, and `http://localhost:8095/readyz` reports whether
planning is configured and the Editoast health endpoint is reachable. Readiness returns
HTTP 503 while planning is unavailable; liveness remains independent of OSRD readiness.
The Compose health check uses `/healthz` so an OSRD startup delay does not mark the
planner process unhealthy; use `/readyz` when checking end-to-end planning readiness.

In the front-enabled development stack, the timetable toolbar opens the planner
through the authenticated gateway at `/ops-intelligence/planner` and passes the
current scenario IDs. Selecting timetable trains pre-fills their IDs as the
eligible retiming set. The retiming policy can also load and search timetable services
by name, route endpoints, rolling stock, and ID; only checked services are eligible to
move, while every unchecked service remains fixed. Direct ID entry remains available
under Advanced. Selecting exactly one existing train also pre-fills its path,
stops, name, and rolling stock as a template. Planners can also search for
operational points by name, choose a rolling stock from OSRD's lightweight catalogue
suggestions, add ordered stops or pass-throughs, and set dwell
times without editing JSON. Planners can select the operating time zone (defaulting
to the browser zone); date-times are converted to UTC, and nonexistent or repeated
daylight-saving wall times are rejected by the browser and repeated local times are
also rejected by the API. The request carries that IANA zone as provenance while UTC
remains the canonical schedule time. Each option shows the requested waypoint names and
its OSRD track-section path, plus simulated departure, destination-arrival, and
travel-time details for each proposed train. Options that retime existing services
also show their OSRD-simulated departure, destination arrival, and travel time, with
the simulated running-time change against each service's source-timetable baseline
when both simulations succeed. It also reports candidate arrival-time change against the
source-snapshot simulated arrival, combining departure retiming and running-time change;
positive values mean later arrival. Hourly cycle options report the same per-service arrival
and running-time deltas, with a partial-coverage count when source or candidate simulation
data is unavailable. The retiming policy lets planners set a global shift cap and optional
lower per-service caps (including zero to protect an otherwise eligible service), and choose
the shift increment used to explore alternatives.
Proposed retimings show train names and schedule IDs. The
form reads the current snapshot binding before it submits a plan request and
presents the OSRD-validated alternatives and proposed edits. Exported review
records include the original request and policy, snapshot binding, all evaluated
candidates, plan counts, and the candidate chosen for review. The results summary shows
how many generated options were evaluated and how many passed OSRD validation, so the
planner can see when the bounded validation budget left candidates unevaluated.
Unselected options are
not labeled as rejected. After a draft scenario is created, the page offers a second
record that captures that action without implying publication or operational acceptance.
A planner can export an option or create a
separate draft scenario containing a copy of the source timetable plus the chosen
service and retimings; before creating the scenario, it reruns simulation and checks
that the draft's exact OSRD conflict set matches the validated option. The source
scenario remains unchanged. The page also runs
directly at `http://localhost:8095/planner` for plan review, but draft creation
and operational-point search require the gateway route so the browser can reach
authenticated Editoast APIs.
For a Docker front-mode stack, start the service and load the gateway route with:

```sh
docker compose -f docker-compose.yml -f docker/docker-compose.front.yml \
  -f ops-intelligence/docker-compose.yml up -d --build ops-intelligence gateway
```

For the repository's host-network front mode, also apply the planner host override so
the gateway can reach the planner on `localhost:8095`:

```sh
docker compose -f docker-compose.yml -f docker/docker-compose.host.yml \
  -f docker/docker-compose.host-front.yml -f ops-intelligence/docker-compose.yml \
  -f ops-intelligence/docker-compose.host.yml up -d --build ops-intelligence gateway front
```

## Validate bounded timetable candidates

The planning route supports delay incidents with an explicit delay and affected
train-schedule IDs. It creates bounded departure-shift candidates, clones the
complete source schedule set into a temporary unpublished timetable, runs
OSRD simulation summaries and conflict detection, deletes the temporary records,
and returns accepted and rejected candidates. Acceptance requires valid pathfinding
and simulation with no conflicts added beyond the source timetable; baseline and new
conflict counts are returned separately. Rejected candidates are ordered by pathfinding,
simulation, new-conflict count, and the selected objective so the closest viable options
are easiest to review. It never edits or publishes the
source timetable. Obtain the current exact binding first with
`GET /v1/operations/snapshots/{infra_id}/{timetable_id}`; its timetable revision
is a SHA-256 over the normalized OSRD train-schedule forms. For `HOURLY` timetables,
positive shift candidates are additionally capped so each paced schedule's start offset
stays below its recurrence interval; schedules with no whole-second room to move remain
unchanged in generated candidates.
Calendar planning can include one OSRD work-schedule group in its conflict checks by
selecting the group ID in the planner. The timetable-conflicts endpoint omits work
schedules unless `work_schedule_group_id` is supplied. Hourly timetables do not accept a
group because date-bounded work schedules do not yet have defined semantics against a
recurring cycle. With no group selected, conflict checks cover trains only.
When a calendar group is selected, both baseline and candidate conflict requests use
that group, and draft creation repeats the same comparison before retaining the draft.
The experimental insertion ranker is skipped for group-aware plans because its training
and offline ranking gate do not cover work-schedule inputs.

From-scratch planning is supported: the source timetable may contain zero train schedules
as long as it has at least one linked schedule set that establishes the timetable type.
Its snapshot revision is bound to that timetable type and the empty schedule list. The
planner then builds each proposal in an isolated OSRD timetable and runs normal pathfinding,
simulation, and conflict validation before returning it.

For requests containing several new service patterns that must be checked together, use
`POST /v1/operations/complete-timetable/plans`. Send `services` as 1–10 complete
`ServiceRequest` objects, all bound to the same snapshot, with unique base train names and
at most 50 total departures. Each request may set `operating_weekdays` to local weekday
numbers (Monday `0` through Sunday `6`). When present, the local wall-clock window repeats
on those days across the inclusive local dates of `operating_window_start` and
`operating_window_end`; the frequency cadence restarts at each selected day. Leave it
empty to preserve one continuous UTC window. Repeated weekday windows must start and end
on the same local day, and nonexistent or ambiguous local times at clock changes are
rejected. `phase_step_seconds`, `max_phase_options_per_service`,
`max_route_alternatives_per_service` (1–5), `max_stop_pattern_options_per_service` (1–16),
`retiming`, and `max_candidates` bound a deterministic joint search across optional stop
patterns, departure phases, OSRD route alternatives, and retiming proposals. Mark an
intermediate waypoint with `optional_stop: true` and a positive `stop_for_seconds` dwell
to let the complete planner compare serving or passing that location. It does not invent
new stop locations. Add `dwell_options_seconds: [30, 60, 90]` to an intermediate waypoint
to compare those dwell values as part of the same bounded stop-pattern search; the chosen
per-stop dwell appears in each pattern result and in the planner review. For an optional
stop, the planner compares passing the stop with serving it at each configured dwell. For
a required stop, it compares only the configured dwell values. Without this field,
`stop_for_seconds` remains fixed. Use `existing_service_dwell_options` to compare dwell
alternatives on named intermediate calls in the source timetable, for example
`[{"train_schedule_id":"123","waypoint_key":"station-key","dwell_seconds":[0,60,120]}]`.
The source dwell is always included as an unchanged option; zero removes that scheduled stop.
Every selected existing-service dwell change is explicit in the candidate, simulated with
the proposed services, and included in the existing-service flow-on and conflict comparison.
The planner carries the selected change into a new OSRD draft only after the planner chooses
that candidate. Set `max_travel_time_seconds` on a service to impose a hard end-to-end
duration limit including scheduled stops. Every departure needs an OSRD-simulated duration
within the limit; a violation or missing simulation time makes the candidate ineligible and
is reported separately from OSRD pathfinding/conflict feasibility. Route alternatives are
generated per requested path leg and combined by
shortest topological length before sampling. Each candidate combines every pattern's
selected route and departures and is validated as one timetable by OSRD;
the response reports the phase and departures for each pattern plus whole-candidate
pathfinding, simulation, and conflict evidence. It compares every existing service's
snapshot simulation time with its candidate simulation time, reports changed or retimed
services, and marks the comparison partial if either timing is unavailable. `total_phase_combinations` reports the
phase space, `total_route_combinations` the joint route choices,
`total_route_phase_combinations` their combined space, `total_stop_pattern_combinations`
the bounded stop-pattern choices, and `total_route_phase_stop_combinations` the combined
search dimension. `retiming_candidate_count` reports edit proposals, and
`total_candidate_combinations` their product. `search_complete: false` means the validation
budget sampled evenly spaced combinations from that product. The selected candidate is the first
OSRD-feasible plan under the selected existing-service retiming objective. Its `retiming`
policy uses the same protected and eligible schedule IDs, per-service caps, shift bounds,
and objective as single-service planning. The response includes explicit edits plus
OSRD-simulated departure, arrival, and running-time effects where available. A patronage
profile can rank complete candidates by joint station-call coverage after feasibility and
new-conflict checks. This transparent coverage proxy is separate from the learned plan
ranker, which remains gated for complete frequency and retiming plans.
The planner page's `Plan a complete set of services` mode builds this request from its main
service form plus up to nine additional patterns. Add patterns with the structured form:
enter each service name, rolling stock, local operating window, and frequency, then choose
its origin and destination from OSRD operational-point search. Add intermediate stops or
pass-throughs and compare fixed or alternative dwell times. For existing services, load
the source timetable's services, select an intermediate call, and compare dwell choices
including stop removal; only services with intermediate path waypoints are eligible, and
the source dwell is always retained as the unchanged option. Raw JSON remains available
for existing integrations. The planner presents each joint
candidate and can materialize a selected feasible option as a separate draft, then checks
that the draft conflict fingerprints still match the validated candidate before keeping it.

The service-planning route (`POST /v1/operations/service-requests/plans`) returns
multiple bounded alternatives for a requested frequency service. The planner samples
first-departure phases using the configured step and maximum phase count, then validates
each complete option through OSRD. If route-phase combinations exceed the validation
budget, it samples evenly across the ordered combinations when more than one slot is
available, preserving route/phase diversity within the budget. Each alternative includes
an `explanation`
summarizing the requested departures, departure phase, route choice, existing-service
retimings, selected objective, and OSRD validation outcome. Feasible options are ranked
first, then by the selected deterministic retiming objective and stable tie-breakers.
The bounded retiming pool covers single-service and pairwise changes, coordinated
same-direction changes, and sampled mixed-direction changes across larger service sets.
Per-service caps are hard bounds within the global retiming cap and affect every generated
single-service, pairwise, and coordinated proposal for that schedule.
The objective can minimize total shift, minimize existing services changed, or minimize
net simulated running-time change across retimed services. A priority-weighted shift
objective is also available in both calendar and hourly service planning: provide a
positive `schedule_priority_weights` value for every eligible schedule, and the planner
minimizes the sum of each absolute shift multiplied by its service weight. Higher weights
make a service more costly to retime. Each result reports that weighted retiming cost.
The simulated running-time objective is scored
only when each retimed service has successful source and candidate simulation summaries;
it is rejected when an eligible paced service has a non-disabled exception that changes
its path, schedule, rolling stock, speed limit, or simulation settings, because those
occurrence-specific running times are not yet compared by the objective. If a candidate
has no complete source-versus-candidate runtime measurement, its runtime objective value
is unknown and the sorter places it after candidates with measured values; that ordering
does not imply the unknown runtime outcome is worse.
The route uses deterministic ordering by default. When an offline-eligible insertion model
is explicitly configured, it can add a separate research suggestion for the narrow
single-departure direct-path case described under the insertion-ranking API; learned
ranking for complete frequency and retiming plans remains disabled until a plan model
passes its frozen holdout gate. When no model is configured, the planner says explicitly
that it is using deterministic ordering and has no AI suggestion. Explanations are derived
from plan data and do not claim predicted passenger benefits. The API route remains
advisory; draft creation is an explicit browser action after the planner selects a
feasible alternative.

### Station patronage demand (initial research capability)

`data_contracts/station-patronage-profile-v1.schema.json` defines a provenance-bearing
profile of average boardings and alightings by OSRD operational point, ISO weekday, and
local-time bucket, bound to an exact OSRD infrastructure ID and version. Single-service
and complete-calendar service-plan requests may include `patronage_profile`; the planner UI accepts the same
profile as a JSON file. Each candidate response includes OSRD-simulated call timestamps
and, when profile station IDs match, a deterministic coverage summary. It reports the
share of supplied boardings/alightings in time buckets touched by at least one candidate
call at the same operational point, for weekdays on which the candidate operates. A
bucket is counted once per active service day even when several trains call in it.
Unmapped calls and mapped calls outside supplied time buckets are reported separately.
Missing simulation times or wholly unavailable station mapping produce an explicit status
and no score. The optional `candidate_order: "station_patronage_coverage"` order ranks
OSRD-feasible options with fewer new conflicts first, then higher combined boarding and
alighting coverage, then the existing retiming objective. Options with unknown coverage
follow scored options. This is deterministic demand-aware ordering, not a learned ranker,
and it identifies the coverage metric as a proxy, not passenger benefit. The experimental
insertion ranker is suppressed whenever a patronage profile or coverage ordering is used,
until it is separately evaluated with patronage inputs.

Hourly-cycle patronage comparison requires explicit `operating_weekdays` (Monday `0`
through Sunday `6`) and a `cycle_start_minute` in the uploaded profile timezone. The
cycle-relative departure phase and OSRD-simulated path-item offsets are projected onto
each selected weekday; calls after midnight are matched against the following weekday.
`candidate_order: "station_patronage_coverage"` applies the same feasibility-first,
conflict-first deterministic coverage order used for calendar plans. Each option reports
coverage and weekday/minute call summaries; the synthetic week anchor is not presented as
a service date. An uploaded hourly profile must match the infrastructure and include at
least one operating weekday. This comparison does not infer origin-destination journeys,
transfers, crowding, willingness to travel, or service reliability. Keep profiles
aggregated and preserve their source, license, observation dates, and timezone; do not add
raw passenger-identifiable records.

The timetable toolbar opens the planner for `CALENDAR` and `HOURLY` scenarios. For hourly
scenarios the planner uses the separate
`POST /v1/operations/hourly-service-requests/plans` endpoint, which explores bounded
cycle-relative phase offsets and OSRD route alternatives for a recurring service. Its
request specifies frequency and time-window durations, an eligible/protected service set,
retiming caps, and an objective. It may also include a versioned patronage profile,
operating weekdays, a local cycle start minute, and a demand-aware candidate order. Its
response returns cycle-relative offsets, route track
sections, explicit existing-service offset changes,
recurring departure counts, simulated runtime when available, and OSRD validation evidence.
Hourly retiming accounts for each service's interval and any occurrence-specific start-time
exceptions. A selected feasible option can be copied into a separate hourly draft scenario;
materialization reruns simulation and conflict checks and compares the draft conflict
fingerprint with the validated candidate. Hourly planning does not yet offer a model
suggestion. OSRD computes an hourly conflict cycle as the
least common multiple of paced-service windows and rejects cycles longer than 24 hours;
candidate validation compares each candidate against the exact baseline conflict multiset.
The calendar response continues to use timestamps and is not reused for hourly offsets.

The repository Compose service defaults `OSRD_EDITOAST_URL` to
`http://gateway:4000/api`; set `OSRD_AUTHORIZATION` when the gateway requires a
service authorization header. Override the URL for other deployments. The adapter
reads the source timetable's linked schedule sets from Editoast and copies every
regular and paced schedule into the isolated candidate. Paced-train exceptions
are recreated against the new schedule IDs, and explicit exception departure
times shift with their parent candidate. The current planner supports delay
incidents only; closures, speed restrictions, and rolling-stock changes need
separate candidate strategies.

`policy.objective` ranks OSRD-validated candidates using `minimize_total_shift`
(default), `minimize_changed_services`, or `minimize_weighted_shift`. For the
weighted objective, provide a positive `schedule_priority_weights` value for
every affected, unprotected schedule; larger values make shifting that service
less desirable. `protected_train_schedule_ids` are excluded from every candidate,
not merely penalized. The response echoes the objective and each candidate's
score, and reports both the proposal-pool size and the number actually validated.
For tighter service-specific retiming rules, `max_shift_seconds_by_schedule`
sets a hard cap (0–7200 seconds) for individual services; omitted services use
`max_departure_shift_seconds`. A zero cap prevents that service from being edited.
Every response also includes a no-change `pass_through_rules_v1` baseline validated
against the exact same snapshot. The baseline is separate from the edited-candidate
budget and is not itself an edit recommendation.
`policy.candidate_validation_order` defaults to `objective_first`. Set it to
`diverse` to spend the validation budget round-robin across distinct sets of edited
services, with lower objective scores first within each set. Returned accepted and
rejected candidates remain sorted by the selected operator objective. This option
is useful when collecting broader candidate outcomes; it does not change hard
constraints or the final OSRD validation gate.

Completed plans are recorded with the full typed request and response, snapshot and
policy, UTC creation time, and SHA-256 digests. Repeating the same request ID and
payload returns the stored result without rerunning OSRD; reusing that ID with a
different payload returns HTTP 409. Retrieve a record with
`GET /v1/operations/plans/{request_id}`. `OSRD_OPS_AUDIT_DB` selects the SQLite
journal path. Compose mounts a persistent named volume. This append-only adapter is
for the fork prototype; production should write authoritative audit records through
Editoast persistence and include the authenticated operator identity.

## Prepare public rail data for forecasting

Dataset research, model-track separation, forecast target/split rules, license notes,
and the exact first training experiment are documented in
[`docs/ai/training-data-and-model-readiness.md`](../docs/ai/training-data-and-model-readiness.md).
RIDE Gold Lite provides a pinned Belgian train/test forecasting dataset; a separate
Swiss OSM infrastructure now provides southern training locations and a held-out northern
topology-transfer benchmark. One day of Swiss Actual data v2 is also locally adapted as
a status-aware disturbance profile;
the full archive and GTFS remain future operational/schedule sources. Neither replaces OSRD
candidate-validation outcomes for the planning feasibility ranker below. The pinned
training split can be fetched locally with SHA-256 verification; interrupted downloads
resume through Hugging Face's Xet cache. RIDE lists its complete
tabular tree at 7.26 GB; the command fetches only the training arrays and schema, not
the held-out arrays or sequential/graph representations:

Install the optional data dependencies, then run the fetcher:

```sh
uv sync --extra rail-data
uv run osrd-ride-fetch-training
```

The downloader uses Hugging Face's Xet-aware client for adaptive parallel transfer,
terminal progress reporting, and resumable Xet cache state. If Xet reports a CAS
response-decoding/reconstruction failure, it retries with Hugging Face's supported
non-Xet HTTP path in a fresh process; if that fails, it tries a resumable HTTPS range
transfer with bounded retries. Pressing Ctrl+C stops the active transfer process and
retains resumable cache state; already verified files remain in place. A Hugging Face
token is not required for this public dataset, but is recommended for a multi-gigabyte
fetch: the Hub warns that anonymous requests have lower limits and may be slower. A
configured `HF_TOKEN` or saved login is used automatically. SHA-256 hashes are verified
before data is accepted. Any `.part` file left by the older single-stream downloader
is not reused by Xet and is removed only after the replacement verifies successfully.

To fit the bounded CPU baseline after acquisition, install the numeric extra and run:

```sh
uv sync --extra rail-data
uv run osrd-ride-train --output-dir data/rail/ride-forecast-model
```

The trainer verifies every pinned input, memory-maps the 5.7 GB feature matrix, samples
up to 20,000 fit rows and 5,000 later rows grouped by service date. Dates after the
default 2024-10-01 cutoff are split chronologically into early-stop, calibration, and
internal-evaluation groups. It optimizes only valid targets and does not load or score
the 2025 holdout. Its report compares each horizon with a train-only mean and a
persistence reference, includes signed bias, and breaks evaluation metrics out by
service month. The model predicts the provider-normalized `future_delay_delta` targets;
it reports target-change metrics and absolute-delay metrics in seconds. For the latter,
it inverts RIDE's signed-square-root z-score on
the future-delay change and adds the latest known delay, so model outputs and the
persistence reference are scored against the same absolute target. See the readiness
plan for the exploratory comparison and its limits.

For the larger bounded comparison, keep the validation rows fixed while varying only
the fit sample and training seed:

```sh
uv run osrd-ride-train --model mlp --max-train-rows 500000 \
  --max-validation-rows 15000 --seed 42 --validation-seed 42 \
  --output-dir data/rail/ride-forecast-mlp-500k
```

Use the same options with `--model linear` for its baseline. `--seed` changes training
sampling and optimization; `--validation-seed` fixes early-stop, calibration, and
evaluation rows so runs remain directly comparable.

The trainer can also fit the first neural baseline, a small feed-forward MLP over the
same RIDE tabular features. It uses train-only feature standardization, two ReLU hidden
layers by default, Adam, L2 regularization, a validity-masked multi-horizon quantile
loss, and p10/p50/p90 outputs:

```sh
uv run osrd-ride-train --model mlp \
  --output-dir data/rail/ride-forecast-mlp \
  --max-train-rows 20000 --max-validation-rows 5000
```

Use the same split, seed, and row limits for the linear comparison. This MLP does not
consume railway topology or event sequences, so it is not the graph-temporal model
specified in the architecture design. Keep it an offline experiment until it beats the
linear and persistence references on a frozen holdout and receives review.
The MLP retains its best early-stop-group masked-pinball-loss checkpoint and stops
after five epochs without improvement. It expands p10–p90 intervals using the separate
calibration group; the final internal evaluation and reserved 2025 holdout stay out of
both fitting and calibration.
Split-conformal coverage is marginal and relies on calibration/evaluation exchangeability;
railway time drift can break that assumption, so monitor coverage by time and corridor.

### GBDT baseline

The design requires the neural forecaster to beat a gradient-boosted reference. The GBDT
baseline fits one histogram GBDT per horizon and p10/p50/p90 quantile on the identical
partitions, applies the same split-conformal interval expansion, and writes a comparable
report. Boosting stages are chosen on the chronological early-stop group, not a random
split. It is a report-only baseline; no servable artifact is exported.

```sh
uv sync --extra rail-data --extra baselines
uv run osrd-ride-train --model mlp --output-dir data/rail/ride-forecast-mlp
uv run osrd-ride-gbdt --output-dir data/rail/ride-forecast-gbdt \
  --compare-report data/rail/ride-forecast-mlp/report.json
```

Pass the same `--seed`, `--validation-seed`, row limits and `--validation-start` to both commands;
`--compare-report` refuses a different dataset revision or seed. Positive
`mlp_improvement_over_gbdt_pct` means the MLP has lower calibrated mean pinball loss.
The RIDE tabular training files must be fully downloaded and hash-verified first.

RIDE Gold Lite also publishes a PyTorch Geometric heterogeneous GNN representation.
The schemas and checksums we retrieved are recorded in
[`data_contracts/ride/gold-lite-gnn-manifest.json`](data_contracts/ride/gold-lite-gnn-manifest.json),
and the graph/data design and acquisition constraints are in the readiness document.
The GNN shards are not downloaded by the tabular fetch command. From the upstream RIDE
repository root, `python -m scripts.dataset.download.download_dataset --target
gold_lite_gnn` is the official route, but fetches the whole GNN component including its
test shard. For controlled training, run this train-only fetch command from
`ops-intelligence/`:

```sh
uv sync --extra rail-data --extra gnn
uv run osrd-ride-fetch-gnn-training
```

It fetches only `gnn/feature_spec.yaml`, `gnn/normalization.yaml`, and the two
`gnn/train/*.pt` shards from revision `e5c1ed1126bca1784eb6ea93816dd6cc89afa544`,
verifies their manifest hashes, and writes a provenance manifest under
`data/rail/ride-gold-lite-gnn-training/`. It never requests the test shard. The first
graph model is an optional offline PyTorch/PyG path. The published graph represents
train/station/event interactions; it is not an OSRD track/switch/signal graph and cannot
replace OSRD validation of a timetable. Do not use the test shard until model selection
is frozen.

The first local graph model is an edge-aware heterogeneous GATv2: it encodes train and
station nodes, exchanges messages over the six directed station/past/future relations,
and predicts normalized delay change on each train-to-future-station edge. The trainer
splits snapshots chronologically by UTC day, keeps each day together, uses the later
20% of days for early stopping, and reports MAE/RMSE in seconds after inverting RIDE's
signed-square-root target normalization. It verifies both shard hashes before loading
and never touches held-out graph files. This is an initial benchmark implementation,
not yet an upstream RIDE GNN reproduction or an OSRD timetable model.

Run it from `ops-intelligence/` after downloading the training graph files:

```sh
uv run osrd-ride-gnn-train
```

The model defaults follow the published Lite scale (128 hidden channels, six layers,
mean relation aggregation, dropout 0.15, and up to 34 epochs) with early stopping. The
trainer prepares reusable 128-snapshot chunks under `data/rail/` and trains from one
chunk at a time; CPU batch size defaults to 4. It writes an atomic recovery checkpoint
every 100 optimizer steps and after each 10 chunks, as well as after every epoch. The
best model and recovery checkpoint are in the ignored `artifacts/` directory. If a run
is interrupted, rerun the same command with `--resume` to continue from the most recent
recovery point:

```sh
uv run osrd-ride-gnn-train --resume
```

For a bounded development run, pass `--max-graphs 64 --epochs 2 --batch-size 2`; use
`--chunk-size` to adjust preprocessing memory and `--checkpoint-every-steps` to change
how often optimizer progress is saved. The serialized train-node tensor has
32 columns while the pinned feature specification lists 31; the current model uses the
stored width. Keep this discrepancy and the fact that the local GATv2 is not an exact
upstream reproduction attached to any result.

After freezing a checkpoint, evaluate it once on the official 2025 GNN test split:

```sh
uv run python -m osrd_ops_intelligence.training.evaluate_ride_gnn
```

The evaluator downloads only the pinned test graph shard if needed, verifies its
published SHA-256, reports overall and per-future-rank MAE/RMSE against a zero-delay-
change reference, and saves `artifacts/ride-gnn-test-report.json`. The first full run
used 15,000 training snapshots; the best checkpoint (epoch 25 of 34) scored 83.28 s MAE
and 205.18 s RMSE over 6,528,133 test targets, versus 97.48 s / 236.87 s for the
reference. Do not use this held-out result to tune the model. It measures RIDE delay
forecasting only; it does not test timetable-change impact prediction or transfer to
another network.

To run a saved linear or MLP artifact on a compatible feature matrix, use the offline
predictor. It verifies the model digest and exact RIDE feature/normalization schemas.
The linear model writes a 2D array; the MLP writes `[rows, 15 horizons, 3 quantiles]`
in provider-normalized target units (p10, p50, p90). Inference sorts crossed quantile
outputs into nondecreasing order; the training report records the pre-sort crossing
rate, pinball losses, and p10–p90 interval coverage by horizon:

```sh
uv run osrd-ride-predict \
  --model data/rail/ride-forecast-mlp/model.json \
  --features data/rail/ride-gold-lite-training/tabular/train/x.npy \
  --feature-schema data/rail/ride-gold-lite-training/tabular/scheme.yaml \
  --normalization data/rail/ride-gold-lite-training/tabular/normalization.yaml \
  --output data/rail/ride-predictions.npy
```

This utility is for offline experimentation and does not connect the forecast to the
planning API or authorize operational changes.

While the RIDE download is pending, generate a tiny synthetic fixture and fit a smoke
MLP with separate training, early-stopping, calibration, and test partitions:

```sh
uv run osrd-ride-synthetic-smoke
```

By default this writes under `data/rail/ride-synthetic-smoke/`: partitioned features,
targets and masks, toy schema/normalization files, uncalibrated and calibrated model
artifacts, predictions, and a report. The calibrator uses split-conformal expansion
from a separate calibration partition; the report evaluates coverage on the separate
synthetic test partition. You can exercise the calibrated predictor with:

```sh
uv run osrd-ride-predict \
  --model data/rail/ride-synthetic-smoke/model.json \
  --features data/rail/ride-synthetic-smoke/features-test.npy \
  --feature-schema data/rail/ride-synthetic-smoke/feature-schema.yaml \
  --normalization data/rail/ride-synthetic-smoke/normalization.yaml \
  --output data/rail/ride-synthetic-smoke/predictions.npy
```

These generated variables are not railway observations or RIDE features. The report
only indicates that the training and inference path ran; do not use its metrics to
assess forecast quality or generalization.

For real experiments, calibrate only with a partition that was not used for model
fitting, checkpoint selection, or final evaluation. Generate that partition's model
predictions first, then run `osrd-ride-calibrate` with those predictions, targets and
validity mask. The resulting artifact records per-horizon interval expansions and a
hash of the calibration labels; evaluate it on a different untouched split.

The synthetic fixture shows the command shape:

```sh
uv run osrd-ride-calibrate \
  --model data/rail/ride-synthetic-smoke/model-uncalibrated.json \
  --predictions data/rail/ride-synthetic-smoke/predictions-calibration.npy \
  --targets data/rail/ride-synthetic-smoke/targets-calibration.npy \
  --mask data/rail/ride-synthetic-smoke/mask-calibration.npy \
  --coverage 0.8 \
  --output data/rail/ride-synthetic-smoke/model-calibrated-cli.json
```

## Prepare planning outcomes for a learned feasibility ranker

Export completed-plan alternatives and OSRD validation labels from the audit
journal into a checksum-bound JSONL dataset with scenario-grouped train/test
assignments:

```sh
uv run osrd-plan-ranker-data \
  --audit-db data/plan-audit.sqlite3 \
  --test-fraction 0.2 \
  --output-dir data/plan-ranker
```

Features describe proposed edits, objective, incident size, and pass-through
baseline validation. The target is strictly whether a candidate passed OSRD
pathfinding, complete simulation, and zero-conflict checks. This is a
feasibility-prioritization dataset, not an operator-preference or service-utility
label. It includes only candidates selected for validation, so it has selection
bias and cannot support unbiased plan-quality claims. Final OSRD validation
remains mandatory.

Planner review exports can be collected separately for future service-plan preference
research. Versioned `osrd-service-plan-review-record/v1` files preserve calendar and
hourly candidate sets and mark one feasible option chosen for review, record that no
candidate passed OSRD validation, or record that the planner declined a feasible option
set as unsuitable.
Each new record identifies the deterministic ranking method that produced its option
order; older records without this field are imported as `unspecified`.
Unselected candidates are not labeled as rejected; a no-suitable-option event applies to
the set as a whole and leaves individual alternatives unclassified. The planner can
optionally record a fixed-category reason for a choice or decline. Converted events
preserve that rationale without treating it as evidence of acceptance or outcome; the
manifest counts choice reasons, decline reasons, and both no-feasible and no-suitable
events separately. If a research model suggestion was shown, the review also preserves
its status, artifact digest, suggested candidate, and note; the event export retains that
provenance without treating the suggestion as an outcome label.
For choices made for review while a model suggestion was available, the manifest also
reports how often the suggested candidate was the one chosen for review. This is a
model-feedback diagnostic only: it does not measure acceptance, service quality, or
operational model performance.
Convert downloaded review JSON files with:

```sh
uv run osrd-service-plan-review-data review-*.json \
  --output-dir data/service-plan-reviews
```

The exporter writes `osrd-ops-service-plan-review-events/v8` rows, identifies the
timetable mode, maps hourly cycle offsets and simulated travel time to the corresponding
candidate features, retains the priority-weighted retiming cost when supplied, and keeps
single-service, hourly, and complete-calendar ranking provenance distinct. Complete
calendar rows add pattern and departure counts, phase spread, selected-route context,
requested frequency, scheduled-stop context, and snapshot-versus-candidate existing-service
effect coverage with positive and negative running-time deltas. These
records remain weak preference events and the manifest stays `training_ready: false`.
Review records preserve the full candidate display order explicitly; older v3 through v7
exported events remain readable by the preference trainer, with absent features treated as
unobserved.

The offline preference trainer now fits a regularized quadratic listwise ranker over
operational trade-offs and, when present, separate boarding/alighting patronage coverage,
coverage availability and mapping-quality signals. Its second-order terms can represent
interactions such as patronage coverage versus retiming cost. Complete-calendar reviews
are a separate `--timetable-mode complete_calendar` cohort, never mixed with single-service
calendar or hourly choices. Their feature set captures joint pattern and phase context.
Use `--require-patronage` to require at least 30 fully profile-backed choice sets in each
train, validation and infrastructure-held-out test split; freeze the split with the same
flag before fitting. Patronage-specific fitting supports calendar and complete-calendar
reviews; hourly weekday-aware matching is not implemented. This only strengthens data
sufficiency checks; it does not promote the research artifact to live planner ranking.
Evaluation still requires the frozen, infrastructure-grouped split and blind planner
choices.

For a prospective model-blind collection session, set
`OSRD_OPS_BLIND_REVIEW_MODE=true` and a stable, study-assigned
`OSRD_OPS_REVIEW_INFRASTRUCTURE_GROUP` on the operations-intelligence service. Use one
group label per independent rail infrastructure across all reviewed timetables; do not
use a timetable ID or a newly assigned database ID as a substitute. This disables
the configured insertion model at the API boundary, displays a blind-review banner, and
accepts submitted reviews only when they carry the blind protocol marker and matching
group label. The Compose service passes both variables through from the shell or `.env`.
The preference
trainer uses only blind-protocol chosen-for-review events; historical, standard, and
model-assisted events remain in the export for coverage reporting but are not training
examples.

Use the dedicated review-pilot overlay to force blind review on and clear any configured
insertion model. Set the same study label for every deployment serving the same physical
network, and use a different stable label only for a genuinely independent network:

```sh
OSRD_OPS_REVIEW_INFRASTRUCTURE_GROUP=network-a \
docker compose -f docker-compose.yml -f ops-intelligence/docker-compose.yml \
  -f ops-intelligence/docker-compose.review-pilot.yml \
  up -d --build ops-intelligence
```

The required Compose interpolation fails before startup if the label is missing. Repeat
the deployment for each participating network, setting its assigned label; do not infer
independence from static dataset names, timetable IDs, or database copies. The collection
overlay does not set credentials or choose an infrastructure on the planners' behalf.

The current preference trainer requires at least eight infrastructure groups and 30
blind multi-option choice sets in every group for each timetable mode. Freeze the
dataset-bound infrastructure assignments before fitting either model. First prepare a
reviewed registry from the actual participating networks; the empty template is at
`data_contracts/service-plan-infrastructure-registry.template.json`. Add one group entry
with `study_infrastructure_group`, a canonical `physical_network_id`, `network_name`,
the observed `source_infrastructure_ids`, an `independence_basis`, `reviewed_by_role`,
and timezone-qualified `reviewed_at_utc`. Confirm that network scopes are physically
independent; distinct labels, deployments, or database IDs alone are insufficient. The
freezer checks exact label coverage, unique physical network IDs, review metadata, and
source infrastructure ID bindings, and stores the registry digest in each split manifest.
Training requires the same registry and digest. This registry is an auditable study
attestation; its text fields do not independently prove the physical facts they describe.

```sh
uv run osrd-service-plan-preference-freeze-split data/service-plan-reviews/review-events.jsonl \
  --timetable-mode calendar \
  --infrastructure-registry data/service-plan-reviews/infrastructure-registry.json \
  --output data/service-plan-reviews/calendar-split.json
uv run osrd-service-plan-preference-freeze-split data/service-plan-reviews/review-events.jsonl \
  --timetable-mode hourly \
  --infrastructure-registry data/service-plan-reviews/infrastructure-registry.json \
  --output data/service-plan-reviews/hourly-split.json
uv run osrd-service-plan-preference-freeze-split data/service-plan-reviews/review-events.jsonl \
  --timetable-mode complete_calendar \
  --infrastructure-registry data/service-plan-reviews/infrastructure-registry.json \
  --output data/service-plan-reviews/complete-calendar-split.json
```

Low-coverage datasets fail closed. The frozen split binds the full event-file SHA-256;
any changed export requires a new split review before fitting. It also freezes top-1
accuracy as primary, mean reciprocal rank and listwise log loss as secondary metrics,
and the displayed-order and uniform-random references. Reports include these measures
for each held-out infrastructure; no preference-fit result is a promotion decision.

To include records explicitly saved in the local plan-audit journal, pass its path
with `--review-db` (and omit downloaded JSON files if they are also present in the
journal, to avoid importing the same request twice):

```sh
uv run osrd-service-plan-review-data \
  --review-db data/plan-audit.sqlite3 \
  --output-dir data/service-plan-reviews
```

In the repository Compose stack, the journal is stored in the
`ops_intelligence_data` volume at `/var/lib/osrd-ops-intelligence/plans.sqlite3`.
Export its reviews inside the running service, then copy the generated dataset to
the host:

```sh
docker exec osrd-ops-intelligence osrd-service-plan-review-data \
  --review-db /var/lib/osrd-ops-intelligence/plans.sqlite3 \
  --output-dir /var/lib/osrd-ops-intelligence/service-plan-reviews
mkdir -p ./data
docker cp osrd-ops-intelligence:/var/lib/osrd-ops-intelligence/service-plan-reviews \
  ./data/service-plan-reviews
```

The export contains only reviews a planner explicitly submitted; downloaded JSON
records already present in that same journal should not also be passed as inputs.

The resulting dataset is explicitly marked `training_ready: false`: a review selection
does not establish acceptance, operational utility, or a realized outcome. It is kept
separate from the feasibility labels above. A research-only listwise ranker can be fit
from multi-option `chosen_for_review` events. Train separate models for calendar and
hourly timetables because their departure-time features have different meanings. Each
model uses only the feasible options shown in its choice sets, splits by infrastructure,
and reports top-1 accuracy, mean reciprocal rank, and listwise log loss against both the
uniform-random reference and current displayed option order, using the first feasible
option as its baseline.
Choices made when an experimental model suggestion was shown are excluded by default to
avoid treating model-influenced selections as independent preference labels; their count
is recorded in the report. Metrics are reported both per split and per infrastructure so
strong pooled performance cannot conceal weak transfer to a held-out network.
Training requires at least eight registered physical infrastructure groups and at least
30 blind multi-option choices per group in that timetable mode, plus minimum event
counts in train, validation, and test. This supports controlled preference experiments;
it does not make the review dataset operationally representative or training-ready.

Fit that experimental preference ranker after exporting the review events:

```sh
uv run osrd-service-plan-preference-train data/service-plan-reviews/review-events.jsonl \
  --timetable-mode calendar \
  --split-manifest data/service-plan-reviews/calendar-split.json \
  --infrastructure-registry data/service-plan-reviews/infrastructure-registry.json \
  --output-dir data/service-plan-calendar-preference-model
uv run osrd-service-plan-preference-train data/service-plan-reviews/review-events.jsonl \
  --timetable-mode hourly \
  --split-manifest data/service-plan-reviews/hourly-split.json \
  --infrastructure-registry data/service-plan-reviews/infrastructure-registry.json \
  --output-dir data/service-plan-hourly-preference-model
uv run osrd-service-plan-preference-train data/service-plan-reviews/review-events.jsonl \
  --timetable-mode complete_calendar \
  --split-manifest data/service-plan-reviews/complete-calendar-split.json \
  --infrastructure-registry data/service-plan-reviews/infrastructure-registry.json \
  --output-dir data/complete-calendar-preference-model
```

The command writes `model.json` and `report.json`, including the input dataset digest,
infrastructure registry digest, infrastructure groups assigned to each split, the
validation-selected training epoch, and held-out metrics. Early stopping uses validation
choice-set log loss; the test split is reserved for final reporting. The artifact is
marked research-only and `live_promotion_eligible: false`; the live planner does not
load it. A later promotion decision needs independent evaluation of planner utility and
operational outcomes, in addition to preference fit, before any model can affect
option ordering.

Train the first feasibility-prioritization model and evaluate it on held-out
scenario groups:

```sh
uv run osrd-plan-ranker-train data/plan-ranker \
  --output-dir data/plan-ranker-model
```

The report compares mean per-scenario OSRD-valid rates in the top 1, 3, and 5
validation slots against ordering by the existing objective score, and includes
an oracle upper bound. Training requires both valid and invalid examples in the
training split. The artifact is an offline experiment; it is not loaded by the
planning API.

## Collect a seeded replay dataset

Start the OSRD stack and import an infrastructure and rolling stock first. The
collector defaults to the local gateway's Editoast route (`http://localhost:4000/api`):

```sh
uv run osrd-replay-collect \
  --infra-id 1 \
  --rolling-stock fast_rolling_stock \
  --start-track TC0 --start-offset 185000 \
  --end-track TD0 --end-offset 24820000 \
  --seed 42 --episodes 10 --trains 8 \
  --output data/replays/run-001
```

The executable checks the pinned OpenAPI artifact's SHA-256 before making API
calls. It creates new unpublished timetable and train-schedule-set records for
each arm of every episode. `manifest.json` captures run parameters and content
hash; `episodes.jsonl` records paired baseline/treatment inputs and outputs, plus
pinned infrastructure and rolling-stock versions. The treatment can apply a
synthetic incident delay, retime one existing service, or add a service. Labels
include newly conflict-involved trains, simulation status, and matched arrival,
travel-time, and energy deltas. An added treatment-only service has no fabricated
baseline arrival target; the GNN masks that node for arrival regression while
retaining it for conflict classification.
The v2 episode schema keeps a treatment view for the initial conflict baseline
while preserving both complete arms under `outputs.paired`.

For scenario diversity within a run, supply multiple headways and an explicit
scenario-family identifier. Collect at least two meaningfully different families
for a group-held-out evaluation:

```sh
uv run osrd-replay-collect \
  --infra-id 1 --rolling-stock fast_rolling_stock \
  --start-track TC0 --start-offset 185000 \
  --end-track TD0 --end-offset 24820000 \
  --seed 51 --episodes 8 --trains 6 \
  --headway-options-s 60 420 600 --scenario-group family-a \
  --output data/replays/family-a
```

JSONL is the initial portable interchange format; promotion to Parquet is deferred
until feature columns and stable nullability are agreed.

The collector is intentionally a data-generation baseline, not an optimizer. Its
departure shifts and destination dwell values are synthetic perturbations, not
observed operational delay labels. Keep temporal/scenario splits grouped by
episode and infrastructure, and do not train on synthetic episodes as if they
were field outcomes.

For a planner-style departure-edit counterfactual, use signed retiming actions.
This keeps the service set fixed while moving one selected train earlier or later;
the paired OSRD simulations label arrival changes and newly introduced conflicts.
Collect separate scenario groups (and, when available, separate infrastructures)
to support held-out evaluation:

```sh
uv run osrd-replay-collect \
  --infra-id 1 --rolling-stock fast_rolling_stock \
  --start-track TC0 --start-offset 185000 \
  --end-track TD0 --end-offset 24820000 \
  --seed 20261022 --episodes 30 --trains 8 \
  --headway-options-s 180 300 420 \
  --intervention-type service-retime --max-retime-s 300 \
  --scenario-group service-retime-a \
  --output data/replays/service-retime-a
```

The default remains `incident-delay` for compatibility with the existing replay
families. Retiming episodes are still synthetic and do not add or remove trains;
they are an initial action-conditioned dataset slice, not timetable construction.

### First signed-retiming GNN benchmark

The first local retiming dataset contains 60 episodes / 480 train nodes across
three headway groups on OSRD `small_infra`. It was generated with 8 trains per
episode, no background departure or dwell perturbations, and a signed retiming
up to 300 seconds. Data and reports remain local under ignored `data/replays/`.

On a held-out `wide` group (20 episodes / 160 trains), arrival-delta MAE was
14.54 s for the GNN and 25.81 s for the linear model, while zero-change was
14.43 s. The held-out group had no new-conflict labels, so it cannot evaluate
conflict detection. On a second split trained on `dense` and tested on `medium`,
the GNN MAE was 14.76 s versus 14.51 s for zero-change. It found all 12 new-
conflict labels, but also flagged all 148 negative trains (F1 0.140, precision
0.075). Its threshold was selected only from the training group's predictions;
this did not improve the false-positive rate. The GNN has not beaten the
no-change arrival estimate or demonstrated useful conflict detection.

The earlier fixed-epoch result above used the previous two-way trainer and is
retained as historical evidence. The current three-way regime needs separate
train, validation, and test groups; this command uses the third retiming group
and will therefore produce a new result:

```sh
uv run osrd-replay-gnn-train \
  data/replays/service-retime-dense \
  data/replays/service-retime-medium \
  data/replays/service-retime-wide \
  --validation-scenario-groups service-retime-medium \
  --test-scenario-groups service-retime-wide \
  --output-dir data/replays/service-retime-gnn-full-regime
```

The benchmark remains an early architecture check: one infrastructure, one
route, one retiming action family, and a single held-out scenario group per run.
Service insertions are covered by the dataset slice below; multiple routes and
networks remain future work.

### Paired service-insertion episodes

Use `--intervention-type service-insertion` to add one treatment-only service at
a sampled departure time between the first and last baseline services. The
collector pairs unchanged services by stable train index, labels conflicts newly
introduced for existing and added trains, and leaves the added train's paired
arrival delta null. Feature schema v2 adds an `is_added_service` node feature.

Collect separate headway groups so one can be held out in full:

```sh
uv run osrd-replay-collect --editoast-url http://localhost:8090 \
  --infra-id 1 --rolling-stock fast_rolling_stock \
  --start-track TC0 --start-offset 185000 \
  --end-track TD0 --end-offset 24820000 \
  --seed 20261031 --episodes 20 --trains 8 \
  --headway-options-s 60 90 120 \
  --max-departure-shift-s 0 --max-dwell-s 0 \
  --intervention-type service-insertion \
  --scenario-group service-insertion-dense \
  --output data/replays/service-insertion-dense
```

The first local batch contains 60 episodes (20 each for dense, medium, and sparse
headways), with nine treatment nodes per episode and 80 newly conflict-involved
train labels across 540 treatment nodes (34 dense, 42 medium, 4 sparse). On a dense held-out group (20
episodes / 180 nodes), the GNN detected all 34 newly conflict-involved trains but
also produced 109 false positives (F1 0.384, precision 0.238, recall 1.0). The
linear classifier on the same split scored F1 0.589 (33 true positives, 45 false
positives, one false negative); the departure-gap reference scored F1 0.395.
Arrival-change MAE was exactly zero for the 160 paired existing services, so this
batch does not teach an arrival-delay effect. These results suggest the graph
classifier is not yet useful, and they come from one route on one infrastructure.

The first route-transfer benchmark collected 60 additional paired insertion
episodes on the demo infrastructure's north–south corridor (`TE1` to `TF1`,
North_station to South_station), with 180–900 s headways and varied departure
and dwell perturbations. Training on the 60 west–east dense, medium, and sparse
episodes and holding out all 60 north–south episodes gave 540 train nodes and
540 test nodes. At the training-selected threshold, the GNN scored F1 0.164
(23 TP, 200 FP, 34 FN; precision 0.103, recall 0.404). The linear classifier
on the same route-held-out split scored F1 0.586 (29 TP, 13 FP, 28 FN), and the
departure-gap reference scored F1 0.484. The GNN's arrival-delta MAE was 0.057
ms, while the zero-change reference was 0 ms, confirming that these insertion
episodes still contain essentially no arrival-change signal. This is a negative
transfer result: the current graph representation and small training set do not
generalize as well as the linear model to this corridor. It is still only a
within-infrastructure synthetic route holdout, not evidence of cross-network
generalization. Local replay data and checkpoints are ignored by Git; the
reproducible reports are under `data/replays/route-holdout-west-to-north-south-expanded/`.

The full validation-selected run used dense and sparse groups for training,
medium for validation, and north–south for the final test. Early stopping selected
epoch 104 of 129. Validation F1 was 0.920, but final route-held-out F1 fell to 0.099
(7 TP, 78 FP, 50 FN; average precision 0.093), compared with F1 0.461 for the
departure-gap baseline and F1 0.261 for the logistic classifier, whose threshold
was also selected on validation (its average precision was 0.586). The GNN's
arrival-delta MAE was 0.092 s against 0.000 s for the zero-change reference. The
strong validation-to-test drop confirms that this small replay model is not ready
to support route-transfer decisions. Full-regime artifacts are in the ignored local directory
`data/replays/full-regime-route-holdout/`.

The same seeded run was repeated into `data/replays/final-model-v2-gated/` after
adding an artifact-level eligibility gate. It reproduced test average precision
0.093 for the GNN versus 0.587 for logistic regression and 0.287 for departure-gap;
arrival-delta MAE was 0.092 s versus 0 s for zero-change. Both model-improvement
checks fail, so the artifact records `offline_candidate_ranking_gate_passed: false`
and `production_promotion_eligible: false`. The prediction CLI surfaces this status;
it does not make an ineligible model appear ready because a checkpoint loads.

### First infrastructure-held-out impact benchmark

To test transfer beyond `small_infra`, the repository's `medium_infra` fixture was
imported into the local OSRD stack as `ops-intelligence-medium` (infra ID 2). A
verified northwest-to-southeast route pilot completed, followed by 60 paired
service-insertion episodes with 8 background trains, 180/300/420 s headways, and
departure and destination-dwell perturbations. Training used only the existing
`small_infra` dense and sparse groups; `service-insertion-medium` was the validation
group; all 60 new-infrastructure episodes were held out for test.

On 540 test nodes with 71 newly conflict-involved trains, the GNN reached average
precision 0.111, below the logistic reference (0.712) and departure-gap reference
(0.231). Arrival-delta MAE was 0.080 s versus 0 s for zero-change. The artifact is
therefore gated off from offline candidate ranking and production. Its prediction
CLI was also exercised on a medium-infrastructure episode after removing labels and
treatment outputs; it returned nine service estimates and
`offline_candidate_ranking_eligible: false`. Local episodes and artifacts are
ignored by Git under `data/replays/medium-northwest-southeast/` and
`data/replays/medium-infra-holdout-gated/`.

### Mixed-route service-insertion GNN

The next data tranche fixed a major coverage gap: previous episodes put every
train on one route. The collector now obtains OSRD block, route, and track-section
paths before either simulation arm, supports multiple `--background-route` values,
and stores a route for each service. The GNN connects train nodes when they are
time-neighbours or share OSRD resources; edge attributes encode normalized
departure gap and shared track, route, and block Jaccard overlap. Training rejects
episodes that do not contain one pre-simulation OSRD resource path per service.
No treatment output or label is used to build the inference graph.

The first multi-route insertion dataset has 640 episodes: 240 for training across two
`medium_infra` corridors and one Y branch, 160 for validation across medium and Y,
and 240 for three untouched test groups. Tests cover a third medium corridor,
`small_infra`, and the opposite Y branch. Each episode has eight background
services and one proposed insertion, with mixed route occupancy and varied
headways/departure/dwell perturbations. Data/manifests are in
`datasets/replay-insertion-v2/`; its calibrated baseline artifact is in
`models/trained/service-insertion-v3/`. Version 3 adds monotone Platt probability
calibration fitted on the two validation groups only. This preserves ranking
order and makes threshold selection use calibrated validation probabilities.

On the held-out medium corridor (720 services, 73 positives), conflict AP was
0.688 versus 0.621 for linear. On held-out `small_infra` (720 services, 42
positives), AP was 0.392 versus 0.253 linear. On the held-out opposite Y branch
(720 services, 73 positives), AP was 0.874 versus 0.770 linear and 0.196 for
departure gap. Each group also exceeded both references by at least 0.05 AP.
After validation-only calibration, aggregate test Brier score was 0.054 versus
0.053 for linear. Group Brier scores were 0.053 (medium), 0.079 (`small_infra`),
and 0.032 (Y); the linear references were 0.050, 0.068, and 0.040 respectively.
Calibration improved the GNN substantially over its raw 0.108 Brier score, but
does not beat the linear reference on every group.
Arrival-delta MAE was
0.81 s versus 0.00 s for predicting no change, so the delay head is ineligible.
Risk thresholds are selected on validation groups from infrastructure IDs 2
and 3; the predictor suppresses threshold flags on `small_infra` (ID 1) while
retaining offline ranking scores. This is a validation calibration step, not
proof of deployment calibration. All fixtures are synthetic; operational
network diversity and measured end-to-end candidate-search benefit remain
before production use.

### Expanded topology transfer run

The expanded data tranche adds four ETCS route/density training cohorts, a held-out
overlapping-routes fixture, and a held-out nested-switches fixture. The v5 run uses
1,120 paired episodes across six OSRD fixture infrastructures: 560 train, 160
validation, and 400 test. Test groups include the original medium, small, and Y
fixtures plus the two newly held-out topologies. Its aggregate conflict AP is 0.767
versus 0.641 for linear, and aggregate calibrated Brier is 0.041 versus 0.045.
However, the strict offline gate requires at least a 0.05 AP gain in every test
group; the nested-switch and opposite-Y gains are only 0.022 and 0.032. The v5
artifact therefore fails the offline ranking gate, and the predictor suppresses
threshold decisions for it. It is not the promoted candidate; v3 remains the last
artifact to pass the older, narrower three-group gate. Neither artifact is approved
for production because the data is synthetic and search benefit has not been measured.
The expanded datasets are in `datasets/replay-insertion-v3/` and
`datasets/replay-insertion-v4/`; the evaluation artifact is in
`models/trained/service-insertion-v5/`.

### Frequency- and stop-conditioned service-insertion model v8

Feature schema v6 adds requested frequency, operating-window duration, number of
departures, stop/dwell summaries, time-of-day encodings, and a slot position for each
inserted train. The predictor retains support for the older v5 feature contract. Six
new paired datasets in `datasets/replay-insertion-v5/` contain 108 frequency-action
episodes across medium, Y-branch, and ETCS fixtures, including an intermediate-stop
cohort. They were captured with pinned OSRD `dev 26d19cb` and include episode checksums
and collection parameters.

The v8 grouped run uses 1,228 episodes: 632 train, 178 validation, and 418 held-out
test across 11/3/6 scenario groups. Test conflict AP is 0.732 versus 0.697 for logistic
regression (Brier 0.0451 versus 0.0457). The per-group promotion gate still fails: its
smallest AP gain is 0.007 on the nested-switch holdout; the frequency holdout gains
0.021. Arrival-delta MAE is 0.412 s versus 0 s for the no-change reference, so that
output remains disabled. The v8 artifact, report, and model card are in
`models/trained/service-insertion-v8/` and remain gated off.

Reproduce the exact grouped v8 fit from the checked-in datasets:

```sh
uv run osrd-replay-gnn-train \
  datasets/replay-insertion-v2/train-medium-northwest \
  datasets/replay-insertion-v2/train-medium-north-south \
  datasets/replay-insertion-v2/train-y-branch-a \
  datasets/replay-insertion-v3/train-etcs-west-southeast \
  datasets/replay-insertion-v3/train-etcs-west-northeast \
  datasets/replay-insertion-v4/train-etcs-central-north \
  datasets/replay-insertion-v4/train-etcs-high-density \
  datasets/replay-insertion-v5/y-branch-frequency-insertion-v1 \
  datasets/replay-insertion-v5/medium-frequency-insertion-v1 \
  datasets/replay-insertion-v5/y-branch-frequency-stops-train-v1 \
  datasets/replay-insertion-v5/etcs-frequency-west-northeast-v1 \
  datasets/replay-insertion-v2/validation-medium-northwest \
  datasets/replay-insertion-v2/validation-y-branch-a \
  datasets/replay-insertion-v5/etcs-frequency-central-north-v1 \
  datasets/replay-insertion-v2/test-medium-southeast \
  datasets/replay-insertion-v2/test-small-infra \
  datasets/replay-insertion-v2/test-y-branch-b \
  datasets/replay-insertion-v3/test-overlapping-routes \
  datasets/replay-insertion-v4/test-nested-switches \
  datasets/replay-insertion-v5/etcs-frequency-insertion-v1 \
  --validation-scenario-groups medium-mixed-corridors-insertion-northwest-v2 \
    y-infra-mixed-branch-insertion-a-validation-v1 etcs-frequency-central-north-validation-v1 \
  --test-scenario-groups medium-mixed-corridors-insertion-southeast-v1 \
    small-infra-mixed-corridor-insertion-v1 y-infra-mixed-branch-insertion-b-v1 \
    overlapping-routes-heldout-transfer-v1 nested-switches-heldout-transfer-v1 \
    etcs-frequency-insertion-v1 \
  --epochs 300 --patience 30 --seed 64 \
  --output-dir models/trained/service-insertion-v8
```

### Advisory insertion-ranking API

`POST /v1/operations/insertion-rankings` accepts up to 16 candidate replay episodes
for one baseline timetable. Each episode must contain the unchanged baseline services,
one appended proposed service, pre-simulation OSRD resource paths, and only the paired
baseline output. Training labels, treatment outputs, and treatment timetable identifiers
are rejected. Configure `OSRD_OPS_INSERTION_MODEL` with an artifact that passes its
offline candidate-ranking gate; the v5 artifact is deliberately rejected. The response
orders candidate IDs by predicted conflict risk to existing services and always marks
the result advisory with `osrd_validation_required: true`.

This endpoint ranks already-constructed insertions. It does not find a route, create
stops/frequency timetables, retime other services, or establish feasibility. Run each
complete candidate through OSRD pathfinding, simulation, and conflict detection before
showing it as feasible. When `OSRD_OPS_INSERTION_MODEL` is configured, the service-plan
route can also attach a research suggestion to feasible options only when the request
contains one departure, a direct origin-destination path, no retimings,
and no route alternatives. It ranks those options against the same unchanged OSRD baseline. The
model suggestion is displayed separately from the deterministic option order and never
changes OSRD feasibility or the planner's default choice. Other service-plan requests
continue with the deterministic ordering. When a suggestion is available, the planner
page provides an explicit “Review research model suggestion” action that selects the
suggested feasible option for inspection and opens the usual review controls; it does
not create a draft scenario or record a choice until the planner takes those actions.
Option cards show both the mean and highest per-service research risk scores, along with
the highest-scored existing schedule, so a mean does not conceal a larger individual score.
A passing historical fixture gate is not evidence of network generalization or
production readiness.

For local development, set `OSRD_OPS_INSERTION_MODEL` to an offline-eligible
artifact path before starting the service. For Compose, set `OSRD_OPS_INSTALL_GNN=true`
and rebuild so the image installs the locked optional Torch CPU and PyG inference dependencies.
Compose passes the model setting through; mount the chosen artifact into the container
and set `OSRD_OPS_INSERTION_MODEL` to its container path. Leave the model unset for
deterministic ordering only. The saved
`models/trained/service-insertion-v3/model.pt` artifact may be enabled for research
review, but its synthetic evaluation does not establish network generalization or
production readiness.

From the repository root, the optional Compose overlay below mounts that local v3
artifact read-only, installs the inference dependencies, and configures the service
to use it:

```sh
docker compose -f docker-compose.yml -f ops-intelligence/docker-compose.yml \
  -f ops-intelligence/docker-compose.research-model.yml \
  up -d --build ops-intelligence
```

This overlay is opt-in and requires the model artifact to exist at the documented
path. The model can suggest among a narrow set of feasible, single-departure direct
route options; it does not rank frequency or retiming plans and is not approved for
operational use. Omit the overlay for deterministic ordering without an AI suggestion.

`service.replay_episode.build_insertion_replay_episode` is a strict adapter for forming
an outcome-free model input from caller-supplied OSRD baseline context. It accepts one
appended service over unchanged baseline schedules and requires baseline simulation
summaries, conflicts, and per-service resource paths. It
rejects retimed or multi-insertion candidates. The service-plan route uses it only for
the narrow single-departure, direct-path case described above.

`build_service_plan_replay_episode` handles the wider plan-model shape: it preserves
baseline service order, verifies that each baseline change is exactly the declared
departure shift, and allows appended proposed departures. It still requires the caller
to provide OSRD simulation and path context and rejects paced schedules. The plan GNN
remains research-only and this adapter is not used for live ranking.

`EditoastOSRDClient.get_schedule_resource_paths` derives per-service resource paths
through OSRD `/pathfinding/blocks` for track-offset and operational-point schedule
locations. Operational-point references remain unchanged in the request; OSRD resolves
their matching parts as waypoint alternatives and returns the path used for model
features. Resource requests retain each schedule's speed-limit tag and
stop-at-end-of-block option. Identical location, rolling-stock, and pathfinding-option
requests are reused within a call. This
supplies one model input; by itself it does not assemble a complete episode.
`get_replay_baseline_context` returns the snapshot-bound source schedule forms, all
successful source simulation summaries, and source conflicts in schedule-ID order;
it rechecks the timetable binding before returning. `build_service_plan_action` maps
the typed plan request and candidate into replay action features. The async
`assemble_service_plan_replay_episode` combines those inputs, candidate schedule forms,
and OSRD resource paths into a complete outcome-free feature episode using only source
simulation/conflict evidence. This adapter is available for offline evaluation and is
not called by the live planning route; the plan GNN remains research-only.
The replay model's `headway_s` was constant at 240 seconds in all 1,934 inspected
service-plan cohort manifests and is set to that training constant by the adapters; it
does not represent live infrastructure headway. A model trained with variable or
infrastructure-derived headway would need a new feature schema and promotion gate.

### Service-request timetable generation and validation

`POST /v1/operations/service-requests/validate` accepts a pinned calendar timetable,
an explicit OSRD path (track offsets or operational-point references), dwell seconds at
path waypoints, a timezone-aware operating window, a rolling stock, and a frequency
from 60 to 3,600 seconds. It expands the window into one to 50 departures, adds them
to a temporary clone of the complete source timetable, and returns OSRD pathfinding,
simulation, and conflict results. Temporary timetable and schedule-set records are
cleaned up; the source timetable is not edited.

The request path is explicit: the planner does not invent unprovided stops or create
recurring multi-day service. In complete-timetable mode, a planner may mark intermediate
dwell stops as optional; bounded search compares subsets jointly with routes and departure
phases, then OSRD validates each whole timetable. This is deterministic stop-pattern
selection, not a learned stop-pattern generator. The companion
`POST /v1/operations/service-requests/plans` accepts an explicit list of retime-eligible
train schedule IDs, protected IDs, a maximum absolute shift, step size, candidate budget,
objective, and a hard maximum number of existing services that may change in any one
candidate (0–12). Eligible IDs only define which services may move; this separate cap
limits coordinated edits. It also searches bounded first-departure phases (default: 60-second
granularity with up to six evenly spaced choices across the feasible frequency cycle),
inserts the requested service, and proposes bounded single-train,
pairwise, and coordinated groups from three eligible trains up to the configured
change cap. Coordinated groups use a shared shift; pairwise candidates also explore
mixed shifts. The combined candidate budget is enforced across phases
and retimings, with at least one insertion-only candidate per phase when budget permits.
Every complete candidate runs through OSRD. Only a candidate
that passes pathfinding and simulation and adds no conflict relative to the exact source
timetable is marked feasible. Existing services are never edited in place. The response
reports every tested delta and its OSRD outcome; rejected options identify affected
services, conflict zones, and the OSRD-reported interval. Search is intentionally bounded and does
not guarantee a globally optimal plan; an empty eligible set validates insertion-only.
`max_route_alternatives` now queries OSRD's infrastructure k-shortest-path endpoint (up to
five alternatives), materializes each selected track sequence as ordered track-offset
waypoints, and validates those waypoints as part of every complete candidate timetable.
Route search accepts track-offset locations and ID-based operational-point references;
operational points expand to their track parts, with searches rejected if a pair of points
would exceed 25 track-part combinations. The endpoint ranks topological path length without
rolling-stock constraints; OSRD's candidate pathfinding and simulation reject paths that
the selected train cannot operate. Alternative paths have
not yet been exercised through the service planner on a multi-route real or OSM corridor.
The v15 training data does contain matched TC0/TC1 platform-route actions on a synthetic
medium fixture, but it finds no route-dependent outcome there. This remains deterministic
candidate generation rather than learned route preference. A request with
`max_route_alternatives=1` keeps the caller's explicit path.

Each response candidate reports its phase offset, route-alternative index, ordered track
sections, and inserted departure times. This is
deterministic candidate generation, not a trained plan generator. The learned
GNN remains gated off and does not choose the route, stop pattern, or retimings. The API
also does not claim user preference or operational suitability; planners must review the
explicit changes and validation evidence.

#### Coupled service-plan training data

`osrd-replay-collect --intervention-type service-plan` writes checksummed v3 paired
episodes containing a frequency insertion plus up to `--max-plan-retimed-services`
signed shifts on existing trains. Feature schema v7 exposes each train's planned shift
and the number of edited existing services; labels remain OSRD-measured per-train arrival
delta, conflict involvement, and simulation status. Six fixture cohorts currently contain
360 episodes (four training groups, one validation group, one held-out test group), plus
a separate three-episode schema pilot. Manifests preserve the OSRD SHA, parameters, and
episode checksum.

Route-conditioned paired episodes use repeated `--insertion-route-via TRACK:OFFSET`
arguments to constrain only the inserted service without creating a stop. Collect each
candidate route with the same seed, baseline route, frequency, profile, and other options;
use distinct `--scenario-group` values. The route evaluator verifies that baseline
schedules and perturbations match and that every evaluated pair belongs to the model's
test partition. The v7 medium-fixture data uses `TC0:550` and `TC1:550`.

The first v7 fit is in
[`models/trained/service-plan-v7-expanded`](models/trained/service-plan-v7-expanded/).
On nested-switches, conflict AP is 0.963 versus 0.943 for logistic regression, below the
0.05 promotion margin; arrival-delta MAE is 22.84 s versus 37.15 s for zero-change. This
does not justify learned candidate ordering: the full dataset is synthetic, only one
network group was held out for final testing, and OSRD remains the feasibility authority.
Reproduce the grouped fit with:

```sh
uv run osrd-replay-gnn-train datasets/replay-service-plan-v1/train-* \
  --validation-scenario-groups overlapping-routes-coupled-retiming-v1 \
  --test-scenario-groups nested-switches-coupled-retiming-v1 \
  --epochs 200 --patience 25 --batch-size 8 --hidden-channels 32 \
  --seed 73 --device cpu --log-every 10 \
  --output-dir models/trained/service-plan-v7-expanded
```

#### Belgian observed timing profile for OSRD replay

Infrabel's monthly raw punctuality files include planned and actual stop arrival/departure
times, signed delay seconds, train numbers, stop IDs, and line IDs. Infrabel publishes
monthly files from 2014 under CC0 with attribution; the official catalogue supplies
month-to-file links. The local data now covers January through December 2024 (about
3.8 GB of raw CSV and 23,114,728 event rows). Raw and normalized data are under ignored
`data/` and are not committed.

Fetch a month, normalize it to Parquet, and build a checksum-bound empirical profile:

```sh
uv run osrd-infrabel-data fetch --months 2024-01 2024-02 2024-03 2024-04 2024-05 2024-06 2024-07 2024-08 2024-09 2024-10 2024-11 2024-12
uv run osrd-infrabel-data normalize --months 2024-01 2024-02 2024-03 2024-04 2024-05 2024-06 2024-07 2024-08 2024-09 2024-10 2024-11 2024-12
uv run osrd-infrabel-data profile --months 2024-01 2024-02 2024-03 2024-04 2024-05 2024-06 2024-07 2024-08 2024-09 2024-10 2024-11 2024-12
uv run osrd-infrabel-data fetch --months 2025-01 2025-02 2025-03 2025-04 2025-05 2025-06 2025-07 2025-08 2025-09 2025-10 2025-11 2025-12
uv run osrd-infrabel-data normalize --months 2025-01 2025-02 2025-03 2025-04 2025-05 2025-06 2025-07 2025-08 2025-09 2025-10 2025-11 2025-12
uv run osrd-infrabel-data profile --months 2025-01 2025-02 2025-03 2025-04 2025-05 2025-06 2025-07 2025-08 2025-09 2025-10 2025-11 2025-12 --output data/rail/infrabel-punctuality/profiles/origin-delay-profile-2025.json
```

The normalizer retains provider delay labels and local planned/actual timestamps, with
clock-derived deltas for quality checks. Across 2024, 99.60% of available departure
labels were within one second of the date-aware clock delta. The annual profile has
1,218,110 first-departure observations and 20,814,142 planned stop-dwell samples. The
replay collector can sample these empirical marginals while OSRD still generates paired
impact labels:

```sh
uv run osrd-replay-collect ... --intervention-type service-plan \
  --infrabel-delay-profile data/rail/infrabel-punctuality/profiles/origin-delay-profile.json
```

Episodes record the profile checksum, observed delay, applied capped shift, and sampled
dwell. The original 160 profile-conditioned episodes use the January-only profile. New
full-year data adds 160 insertion episodes and 160 intermediate-stop episodes across four
fixtures, followed by stop/frequency variations at 120-, 180-, 300-, and 600-second
frequencies, including nested-switches. The current fit is recorded in
[`models/trained/service-plan-v11-full-corpus`](models/trained/service-plan-v11-full-corpus/):
1,145 train / 20 validation / 120 test episodes. On three synthetic holdout groups,
conflict AP is 0.963 versus 0.924 for logistic regression and Brier score is 0.102 versus
0.141. Arrival-delta MAE is 13.58 s versus 80.26 s for zero-change. Per-group AP gains are
0.070 on sparse medium traffic, 0.027 on nested-switches, and 0.010 on the tiny fixture.
The required 0.05 margin still fails, so ranking decisions remain disabled. Arrival
estimates remain scoped to the validation infrastructure and are suppressed on all test
fixtures. The annual timing profile does not map Belgian line or stop IDs to OSRD topology
or make simulated impacts real-world labels. OSRD remains the feasibility authority.
A seed-42 refit on the same split scores aggregate AP 0.966 vs 0.924 logistic and group
gains of +0.067, +0.014, and +0.024; it confirms the per-group +0.05 gate still fails.
Because it reuses the same holdout cases, this is a seed-sensitivity check, not independent
generalization evidence. Its artifact is
[`models/trained/service-plan-v11-full-corpus-seed42`](models/trained/service-plan-v11-full-corpus-seed42/).

The comparable phase-feature fit is
[`models/trained/service-plan-v12-phase-choice-comparable`](models/trained/service-plan-v12-phase-choice-comparable/).
It adds four fixed-phase heldout cohorts to the 62 prior source cohorts, preserves the
original validation cohort, and holds all prior test cohorts and all new cohorts out
(1,325 train / 20 validation / 780 test episodes; 37 / 1 / 20 scenario groups). Aggregate
conflict AP is 0.979 versus 0.954 logistic, but the minimum per-group gain is -0.036, so
the +0.05 gate fails. On 40 matched pairs each, pairwise objective accuracy was 53.75%
at 120-second and 51.25% at 600-second frequency (half credit for actual ties). Mean
new-conflicted existing services were 2.075 versus 2.113 for random choice at 120-second
frequency and 3.300 versus 3.313 at 600 seconds; paired positive arrival delay did not
change. These results do not show dependable planner benefit. See the evaluation JSON
files in the artifact directory and the source manifests under
`datasets/replay-service-plan-v4/medium-infra-phase-choice*-heldout-v1/`.

The next refit added four tiny-topology training cohorts (160 episodes across 120/240/
420/600-second frequencies and 60/90/120/180-second stop dwell). The latest refit adds
two TC1 alternate-platform cohorts (80 episodes) while keeping the same validation and
all 20 test groups unchanged. Artifact:
[`models/trained/service-plan-v12-route-topology-expansion`](models/trained/service-plan-v12-route-topology-expansion/).
It has 1,565 train / 20 validation / 780 test episodes. Aggregate heldout AP is 0.987
versus 0.948 logistic and Brier is 0.061 versus 0.121. Tiny-fixture AP gain is +0.005;
the minimum group gain is +0.003, so the +0.05 gate remains false. Phase-choice accuracy
is 56.25% at 120-second and 51.25% at 600-second frequency. This still does not justify
planner ranking.

#### 2025 Infrabel-conditioned expansion

The local ignored data now contains all twelve 2025 Infrabel passenger punctuality files
(22,427,357 event rows), their normalized Parquet files, and the profile
`data/rail/infrabel-punctuality/profiles/origin-delay-profile-2025.json` (1,178,119
train-origin delay samples). Four new 40-episode paired service-plan cohorts use that
profile on the small, Y-branch, ETCS, and nested-switch OSRD fixtures. Their manifests
record the profile checksum and all episode-file checksums under
`datasets/replay-service-plan-v6/`.

The refit artifact is
[`models/trained/service-plan-v13-infrabel-2025-expanded`](models/trained/service-plan-v13-infrabel-2025-expanded/).
It preserves the previous validation group and all 20 held-out groups (1,725 train / 20
validation / 780 test episodes). Test conflict AP is 0.9862 versus 0.9530 for logistic;
Brier score is 0.0693 versus 0.1155. Arrival MAE is 3.67 s versus 46.45 s for zero-change.
The minimum test-group AP gain is only 0.0018, below the unchanged +0.05 gate, so the
model remains unavailable for candidate ranking. The 2025 observed delays only condition
synthetic OSRD scenarios; they are not real-network counterfactual labels. Re-evaluating
the frozen v13 model on the existing phase pairs gives 56.25% choice accuracy at 120 s
(2.05 mean conflicts versus 2.113 for random) and 51.25% at 600 s (3.30 versus 3.313).
This does not show useful route/phase search benefit.

#### OSM-derived Belgian topology benchmark

To add topology beyond the synthetic fixtures, the Geofabrik Belgium OSM extract dated
2026-10-15 was downloaded and verified against its published MD5. The OSM-derived data
is ODbL and attributed to OpenStreetMap contributors; it is not an authoritative Infrabel
infrastructure. OSRD's converter inferred 15,571 track sections, 13,479 switches, and
542 operational points, then generated signals. Because switch connectivity is inferred
from geometry and generated signals are synthetic, this is a research benchmark only.
The source checksums and conversion details are in the ignored
`data/rail/osm-belgium/source-manifest.json`.

The converter emitted two exact duplicate operational-point identities. The
`osrd-osm-railjson-normalize` command removes exact duplicates with matching codes,
name, and track-position set; unresolved collisions fail closed by default. For a
research-only import where OSRD's domestic or UIC-plus-secondary unique indexes collide,
`--disambiguate-conflicting-osrd-identities` deterministically suffixes the local codes
with the UIC or OSM ID. PLC collisions still fail closed. The normalized 540-point
infrastructure imported locally as OSRD infra 13. Same-track pathfinding succeeded; a
cross-track Brussels corridor request timed out after 30 seconds, and Editoast restarted.
The topological k-shortest endpoint and full block pathfinding/simulation succeeded on a
short adjacent-track path (`1122923669-0` to `116000117-0`).
After confirming same-track simulation, one narrow 40-episode cohort was collected on
track `116000117-0` (offsets 10–400 m), with insertion timings conditioned on the 2025
Infrabel delay profile. This is a single-track topology smoke cohort; it does not validate
cross-track corridor operations or transfer to the Belgian network. A separate 40-episode
adjacent-track cohort is held out in v16; these short paths do not establish broad corridor
transfer.

Generate the OSM-cropped RailJSON using either `--bounds WEST SOUTH EAST NORTH` or
`--route-start TRACK --route-end TRACK`; a route crop uses switch-port connectivity from
the source graph and includes complete switch junctions. For example:

```sh
uv run osrd-osm-railjson-crop \
  data/rail/osm-belgium/belgium-261015-deduplicated.railjson \
  data/rail/osm-belgium/brussels-midi-nord-route-crop.railjson \
  --route-start 116000117-0 --route-end 116505098-0
```

A bounded Midi–Nord crop (711 tracks, 515 switches; bounds and checksum are recorded in
`data/rail/osm-belgium/source-manifest.json`) also imported successfully as local infra 14.
Shortest-path search between Midi and Nord again exceeded 30 seconds, after which
Editoast stopped responding. Geographic cropping alone did not resolve the route-search
bottleneck. A route-specific crop command (`osrd-osm-railjson-crop --route-start
116000117-0 --route-end 116505098-0`) retained 55 tracks and 34 switches and imported as
infra 15/16. OSRD can route only the initial few sections and returns no Midi–Nord path;
the crop is not training data because no complete path or simulation labels were produced.

The v14 refit in
[`models/trained/service-plan-v14-osm-brussels-midi`](models/trained/service-plan-v14-osm-brussels-midi/)
adds those 40 episodes to training while preserving the v13 validation and all 20 test
groups (1,765 train / 20 validation / 780 test). Test conflict AP is 0.9875 versus 0.9528
for logistic, and Brier is 0.0668 versus 0.1158. Arrival-delta MAE is 4.36 s versus
46.45 s for zero-change. The minimum per-group AP gain is only 0.0024, so the +0.05
promotion gate remains false. The frozen v14 phase-choice benchmark obtains 56.25%
accuracy at 120 s (2.05 mean conflicts versus 2.113 for random) and 48.75% at 600 s
(3.325 versus 3.313); this does not show reliable search benefit. The test groups remain
synthetic OSRD fixtures, not held-out Belgian topology, and candidate ranking stays
disabled.

V16 adds the 40-episode adjacent-track OSM cohort as a new held-out group while preserving
the previous validation group and all 22 existing test groups (1,845 train / 20 validation
/ 900 test, across 50 / 1 / 23 groups). On the OSM group, conflict AP is 1.000 for both
the GNN and logistic reference; Brier is 0.197 versus 0.561. Arrival-delta MAE is 72.86 s
versus 20.44 s for zero-change, so the arrival head fails to transfer. Minimum test-group
AP improvement is zero, keeping the ranking gate false. Artifact:
[`models/trained/service-plan-v16-osm-adjacent-holdout`](models/trained/service-plan-v16-osm-adjacent-holdout/).

#### Geographic OSM training and phase-choice evaluation

**OSRD path-coordinate units:** timetable locations and `/pathfinding/blocks` use
millimetres, while the separate `/infra/{id}/pathfinding/?number=` k-shortest endpoint
uses metres for both request positions and returned `track_ranges`. The planner adapter
converts in each direction before producing timetable waypoints.

**OSM offset units:** RailJSON `track_sections[].length` and operational-point `position`
are metres. Editoast `track_offset` locations and path ranges are millimetres. The cohort
collector converts all OSM-derived offsets at that boundary, records
`track_offset_unit: "mm"`, and places station stops by passenger operational-point ID.
Do not train or claim route-extent transfer from older OSM switch cohorts that lack this
unit marker; those datasets were collected before the metre-to-millimetre conversion was
fixed. Their outcomes are still accurate for the paths OSRD actually simulated, but those
paths do not match the original OSM metre offsets.

`osrd-osm-replay-cohort` selects switch-connected routes in distinct 0.25-degree cells
from the normalized Belgium OSM RailJSON. The v17 training set contains 480 paired
simulation episodes from 12 southern locations. The geographic test set contains 160
episodes from four northern locations (`latitude >= 50.7`); all routes passed OSRD
pathfinding and simulation, with no failed trains. The four north test locations were
held out by scenario group. These are OSM-derived switch corridors inside one Belgian
infrastructure, not four independent operator networks.

The collector also accepts matched route-waypoint actions through
`--route-via-options`. Include `none` and one or more comma-separated ordered
`TRACK:OFFSET_MM` waypoint sets; every option uses the same scenario group and seed so
the replay builder forms matched action contexts. These are constrained waypoint
requests, not independently discovered route alternatives; OSRD pathfinding still
determines and validates the resulting path. For example:

```sh
uv run osrd-osm-replay-cohort data/rail/osm-belgium/belgium-261015-deduplicated.railjson \
  --collect --output-root datasets/osm-route-waypoint-training-v1 \
  --partition train --limit 12 --episodes 40 --trains 8 \
  --route-via-options none '1222277443-1:2848499,294971646-0:1034704' \
  --no-delay-profile --seed 20261050
```

V17 has 2,325 train / 20 validation / 1,060 test episodes across 62 / 1 / 27 groups.
Overall test conflict AP is 0.9832 versus 0.8789 for logistic regression; Brier is 0.0687
versus 0.1309. Arrival-delta MAE is 4.71 s versus 42.35 s for zero-change. On the four
northern locations, group AP gains over logistic are 0.012–0.064. The minimum improvement
across all held-out groups remains zero, so candidate ranking is still disabled. Artifact:
[`models/trained/service-plan-v17-osm-geographic`](models/trained/service-plan-v17-osm-geographic/).

Phase-choice data initially exposed weak action ranking: on 160 matched 0s-versus-180s
pairs, v19 was correct on 28.75% and selected schedules averaged 3.138 new conflicts
versus 2.909 for random. V20 added 480 matched phase-180 training episodes; its 0s/180s
test on four northern locations reached 66.25% pairwise objective accuracy and 2.850
conflicts versus 3.038 for random.

V21 added 960 training episodes at 60s and 120s on the same 12 southern switches while
preserving v20's exact validation and test groups. V22 then kept those training episodes
and added a fresh 60s-versus-120s test on four previously unused northern switches. Its
split is 3,765 train / 20 validation / 2,340 test episodes across 98 / 1 / 55 groups.
Overall test conflict AP is 0.9693 versus 0.8819 logistic; Brier is 0.0793 versus 0.1342,
and arrival MAE is 4.58 s versus 38.87 s for zero-change. On the fresh 160-pair phase
holdout, model-selected schedules average 2.069 new conflicts versus 2.288 for random,
with 66.56% pairwise objective accuracy. Three locations improve and one ties. The
minimum test-group AP gain is zero, so the promotion gate remains false and no planner
ranking/pruning is enabled. Artifacts and per-location reports are in
[`models/trained/service-plan-v22-multiphase-heldout`](models/trained/service-plan-v22-multiphase-heldout/).

The OSM cohort collector now accepts `--frequency-options-s` and writes distinct
frequency-specific scenario groups. V23 adds 960 training episodes at 120s and 600s on
the 12 southern switches, then tests 160 matched 120s-versus-600s pairs on four more
unused northern switches. V23 has 4,725 train / 20 validation / 2,660 test episodes
across 122 / 1 / 63 groups. Test conflict AP is 0.9653 versus 0.8925 logistic; Brier is
0.0856 versus 0.1287. On the fresh frequency holdout, pairwise objective accuracy is
90.0%; model-selected candidates average 2.675 new conflicts versus 3.241 for random.
All four held-out locations improve, while positive arrival delay is unchanged. However,
the minimum per-group AP gain is -0.0154, so the overall promotion gate remains false and
the model stays out of planner ranking. Evaluation reports are in
[`models/trained/service-plan-v23-frequency-generalization`](models/trained/service-plan-v23-frequency-generalization/).

#### Independent Swiss topology evaluation

The [Geofabrik Switzerland OSM snapshot](https://download.geofabrik.de/europe/switzerland.html)
was downloaded for data through 2026-10-15, converted with OSRD's `osm_to_railjson`
converter, and imported as infra 17. The normalized RailJSON has 19,313 tracks, 17,694
switches, and 790 operational points; checksums and ODbL attribution are in
[`data/rail/osm-switzerland/source-manifest.json`](data/rail/osm-switzerland/source-manifest.json).
The frozen independent test set has 320 simulated episodes from four northern Swiss
switch locations, with 160 matched 120s/600s frequency pairs. V24 and v25 use no Swiss
episodes for fitting; the v25 test additionally uses the one-day status-filtered Swiss
Actual profile.

V24 trains on the Belgian corpus and adds the Swiss groups only to its test split (4,725
train / 20 validation / 2,980 test episodes; 122 / 1 / 71 groups). On the 160 Swiss pairs,
the model chooses the lower-conflict frequency 80% of the time and averages 1.831 new
conflicts versus 2.238 for random. Arrival delay is unchanged. Pooled test conflict AP is
0.9606 versus 0.8899 logistic, but the minimum test-group AP gain is -0.0176; the ranking
gate remains false, and predictions are not operationally eligible for Swiss infra 17.
Per-switch results are in
[`models/trained/service-plan-v24-swiss-network-holdout/swiss-frequency-choice-summary.json`](models/trained/service-plan-v24-swiss-network-holdout/swiss-frequency-choice-summary.json).

#### Swiss Actual data conditioning

The [Swiss Actual data v2](https://data.opentransportdata.swiss/en/dataset/ist-daten-v2)
file for 2026-10-14 was acquired (661,083,046 bytes; SHA-256 recorded in
`data/rail/swiss-actual-v2/source-manifest.json`). The adapter groups events by service
date, operator, and journey; selects the earliest scheduled departure at a Swiss stop;
and uses only `AB_PROGNOSE_STATUS=REAL` for delay samples. `PROGNOSE`, `GESCHAETZT`,
`UNBEKANNT`, empty statuses, cancelled runs, and non-Swiss stops do not become delay
labels. Scheduled arrival/departure differences supply dwell context. The resulting
profile contains 146,533 effective first-departure delay samples and 2,250,129 scheduled
dwell samples. The publisher notes that its wider records can be forecasts or estimates,
so this profile is deliberately status-filtered. The 2026-10-12 file was excluded after
the publisher reported malformed rows.

V25 adds 320 Swiss-profile-conditioned episodes only to the test set while preserving the
Belgian training data (4,725 train / 20 validation / 3,300 test episodes). On the 160
Swiss 120s/600s pairs, the model selects the lower-conflict frequency 92.5% of the time
and averages 2.950 conflicts versus 3.634 for random; positive arrival delay is unchanged.
This tests simulated impacts under one day's Swiss operating-delay marginals, not
counterfactual field outcomes. The promotion gate remains false. See
[`models/trained/service-plan-v25-swiss-actual-conditioned/swiss-profile-frequency-choice-summary.json`](models/trained/service-plan-v25-swiss-actual-conditioned/swiss-profile-frequency-choice-summary.json)
and the profile manifest in `data/rail/swiss-actual-v2/`.

V26 adds 960 profile-conditioned replay episodes from 12 geographically separated
southern Swiss switch locations at 120s and 600s. One southern location is reserved for
validation; all prior test groups, including the four northern Swiss locations, remain
frozen. The fit used 5,605 training / 100 validation / 3,300 test episodes across
144 / 3 / 71 groups. Test conflict AP is 0.978 versus 0.903 for logistic and Brier is
0.0662 versus 0.1278; arrival MAE is 3.27 s versus 38.48 s for zero-change. The smallest
per-group AP difference is -0.496 on the Midi–Nord adjacent-track benchmark, so the
promotion gate fails. On the frozen profile-conditioned Swiss test, v26 selects the
lower-conflict frequency in 92.5% of 160 pairs, averaging 2.950 conflicts versus 3.634
for random, unchanged from v25. The replay outcomes are simulated counterfactual labels;
planner ranking remains disabled. See
`models/trained/service-plan-v26-swiss-multinetwork/` for the fit report and pair results.

Two additional official Actual v2 archives are now processed locally: August 2025 and
March 2026. The bounded profile covers 62 service days, with 7,708,831 REAL first-stop
departure samples in the source population; the replay profile retains 250,000 delays
and 250,000 dwell values. A second matched Swiss training seed uses this seasonal profile
for v27, while the northern test remains on the original one-day profile. V27 completed
with 6,485 train / 180 validation / 3,300 test episodes across 144 / 3 / 71 groups. Test
conflict AP is 0.978 versus 0.907 for logistic, Brier 0.0686 versus 0.1266, and arrival
MAE 2.66 s versus 38.48 s for zero-change. The minimum per-group AP difference is
-0.0048, below the +0.05 ranking gate. On the frozen Swiss holdout it selects the
lower-conflict option in 92.5% of 160 pairs (2.950 conflicts versus 3.634 random),
unchanged from v25/v26. The new profile improves stability across test groups but does
not change this choice score. Source ZIP and daily member digests are in
`data/rail/swiss-actual-v2/source-manifest.json`.

**Modal-data correction:** Swiss Actual v2 contains multiple transport products. The
v25-v27 Swiss profiles did not filter `PRODUKT_ID`, so their Swiss-conditioned replay
contexts mixed rail and non-rail delays; those results are historical and are not rail-only
benchmarks. The rebuilt 2026-10-14 rail profile filters `Zug` and contains 14,026 `REAL`
first-departure samples (2,423,561 non-rail rows excluded). The prior artifacts remain
for audit; v28 excludes their profile-conditioned cohorts.

**Rail-only multi-frequency fit (v28, 2026-10-17):** the adapter scanned all 15 Swiss
Actual v2 monthly archives from 2025-07-13 through 2026-09-30 (445 daily members,
1,054,186,238 input rows). It filtered `PRODUKT_ID=Zug`, excluding 974,785,755 non-rail
rows, and built a bounded 250,000-delay / 250,000-dwell profile from 6,117,227 first-stop
`REAL` departures. This conditions synthetic OSRD training episodes; it is not a source of
counterfactual labels. Eight southern Swiss switch sites contribute 1,680 training and 240
validation episodes across 120/180/240/300/420/600-second service frequencies. Four
northern sites contribute 960 held-out episodes using a separate rail-only profile from
2026-10-14. The train/test split also retains the unaffected Belgian and fixture cohorts,
for 6,405/260/3,620 train/validation/test episodes overall.

V28 selected epoch 35 of 60. Aggregate held-out conflict F1 is 0.900 (AP 0.978 versus
0.929 for logistic regression); arrival-delta MAE is 2.55 seconds versus 38.75 seconds
for a zero-change reference. On 160 matched northern contexts and 2,400 frequency-pair
comparisons, pairwise objective accuracy is 75.4%; the six-way choice matches the
simulator’s lexicographic conflict/delay optimum in 93.75% of contexts. It averages 2.863
new conflicts versus 3.500 for a fixed 300-second choice, but ties a fixed 600-second
choice at 2.863 conflicts and 93.75% exact-match rate. Thus this experiment does not show
benefit over the simplest low-frequency policy; it also does not encode passenger demand
or a planner’s required frequency. The service-side ranking gate remains false because
the per-group AP margin fails (minimum -0.0089); no production promotion is warranted.

Profiles, source/member checksums, and row-quality counts are in
`data/rail/swiss-actual-v2/source-manifest.json`. The model, complete split report, and
held-out frequency-grid results are in
`models/trained/service-plan-v28-swiss-railonly-fullmultifrequency/`. The current GNN is
an action-impact estimator; it does not synthesize full multi-line timetables or decide
which request constraints may be relaxed.

Reproduce the frozen Swiss frequency-grid evaluation with:

```sh
uv run osrd-replay-frequency-grid-eval \
  --model models/trained/service-plan-v28-swiss-railonly-fullmultifrequency/model.pt \
  --data-root datasets/replay-service-plan-v7/switzerland-osm-frequency-choice-railonly-one-day-v1 \
  --switch-id 60621720 --switch-id 11308554127 \
  --switch-id 256547633 --switch-id 11734772012 \
  --frequency-options-s 120 180 240 300 420 600 \
  --fixed-baseline-frequency-s 300 --fixed-baseline-frequency-s 600 \
  --output models/trained/service-plan-v28-swiss-railonly-fullmultifrequency/swiss-frequency-grid-evaluation.json
```

Reproduce the switch-level acquisition from the running local OSRD stack (infra 13):

```sh
uv run osrd-osm-replay-cohort data/rail/osm-belgium/belgium-261015-deduplicated.railjson \
  --collect --output-root datasets/replay-service-plan-v7/belgium-osm-switch-geographic-train-v1 \
  --partition train --limit 12 --episodes 40 --trains 8 --seed 20261020

uv run osrd-osm-replay-cohort data/rail/osm-belgium/belgium-261015-deduplicated.railjson \
  --collect --output-root datasets/replay-service-plan-v7/belgium-osm-switch-phase-training-v1 \
  --partition train --limit 12 --episodes 40 --trains 8 --phase-options-s 180 --seed 20261020

# Additional training-only phase values; 240s is invalid at the 240s minimum headway.
uv run osrd-osm-replay-cohort data/rail/osm-belgium/belgium-261015-deduplicated.railjson \
  --collect --output-root datasets/replay-service-plan-v7/belgium-osm-switch-intermediate-phase-training-v1 \
  --partition train --limit 12 --episodes 40 --trains 8 \
  --phase-options-s 60 120 --seed 20261060

# Training-only frequency alternatives at 120s and 600s.
uv run osrd-osm-replay-cohort data/rail/osm-belgium/belgium-261015-deduplicated.railjson \
  --collect --output-root datasets/replay-service-plan-v7/belgium-osm-switch-frequency-training-v1 \
  --partition train --limit 12 --episodes 40 --trains 8 \
  --frequency-options-s 120 600 --phase-options-s 0 --seed 20261120

# Fresh 60s/120s northern evaluation (exclude all previously used northern switch groups).
uv run osrd-osm-replay-cohort data/rail/osm-belgium/belgium-261015-deduplicated.railjson \
  --collect --output-root datasets/replay-service-plan-v7/belgium-osm-phase-choice-north-multiphase-fresh-v1 \
  --partition test --limit 4 --episodes 40 --trains 8 --phase-options-s 60 120 \
  --seed 20261070 --exclude-switch-ids 3070525079 3098571334 3696834806 4628503095 \
  5298974073 5298974077 5550976753 7006569903

# Fresh, matched frequency-choice evaluation data at 120s versus 600s.
uv run osrd-osm-replay-cohort data/rail/osm-belgium/belgium-261015-deduplicated.railjson \
  --collect --output-root datasets/replay-service-plan-v7/belgium-osm-frequency-choice-north-fresh-v1 \
  --partition test --limit 4 --episodes 40 --trains 8 \
  --frequency-options-s 120 600 --phase-options-s 0 --seed 20261130 \
  --exclude-switch-ids 26022800 500183482 5129374872 9252630098 \
  3070525079 3098571334 3696834806 4628503095 5298974073 5298974077 \
  5550976753 7006569903

# Independent Swiss northern topology holdout (infra 17).
uv run osrd-osm-replay-cohort data/rail/osm-switzerland/switzerland-261015-normalized.railjson \
  --collect --output-root datasets/replay-service-plan-v7/switzerland-osm-frequency-choice-heldout-v2 \
  --partition test --test-latitude 46.8 --limit 4 --episodes 40 --trains 8 \
  --infra-id 17 --network-name switzerland-osm --frequency-options-s 120 600 \
  --phase-options-s 0 --seed 20261220 --no-delay-profile

# Swiss southern frequency-choice training (one southern site reserved for validation).
uv run osrd-osm-replay-cohort data/rail/osm-switzerland/switzerland-261015-normalized.railjson \
  --collect --output-root datasets/replay-service-plan-v7/switzerland-osm-frequency-training-v1 \
  --partition train --test-latitude 46.8 --limit 12 --episodes 40 --trains 8 \
  --infra-id 17 --network-name switzerland-osm --frequency-options-s 120 600 \
  --phase-options-s 0 --seed 20261250 \
  --delay-profile data/rail/swiss-actual-v2/profiles/origin-departure-delay-profile-2026-10-14.json

# Build the local Swiss delay/dwell profile from the downloaded 2026-10-14 v2 file.
uv run osrd-swiss-actual-profile \
  data/rail/swiss-actual-v2/2026-10-14_istdaten.csv \
  --output data/rail/swiss-actual-v2/profiles/origin-departure-delay-profile-2026-10-14.json

# Build a bounded profile from archived monthly v2 ZIPs; one daily CSV is processed at a time.
uv run osrd-swiss-actual-profile --monthly-archives \
  data/rail/swiss-actual-v2/archive/ist-daten-v2-2025-08.zip \
  data/rail/swiss-actual-v2/archive/ist-daten-v2-2026-03.zip \
  --output data/rail/swiss-actual-v2/profiles/origin-departure-delay-profile-2025-08-2026-03.json
```

#### Route-action data and v15 evaluation

The collector now records a route-conditioned treatment with `--insertion-route-via`.
On the medium synthetic OSRD fixture, two matched 40-episode route pairs were collected
with the 2025 Infrabel profile: seed 20261094 is training data (TC0 and TC1 cohorts), and
seed 20261093 is reserved for test (TC0 and TC1 cohorts). Across each pair, all episode
keys, baselines, and background perturbations match; the inserted service's OSRD resource
graphs differ. The test pair's 40 outcomes are tied: neither route changes per-train
conflict or arrival labels.

V15 trains on 1,845 episodes, validates on 20, and tests on 860 across 50/1/22 scenario
groups. Aggregate test conflict AP is 0.9844 versus 0.9527 for logistic regression and
Brier is 0.0670 versus 0.1153. The minimum per-group AP gain is only 0.00094, so ranking
remains disabled. On the 40 held-out TC0/TC1 pairs, route-choice accuracy is 50%; both
routes average 1.85 newly conflicted existing services and 55.9 seconds of positive
arrival delay. The evaluation also reports how many pairs have different OSRD outcomes
and the model's accuracy restricted to those informative pairs. This fixture has none,
so the 50% tie-adjusted score shows no route-choice skill. The model provides no
route-choice benefit for this fixture. Artifact and
reports: [`models/trained/service-plan-v15-route-choice`](models/trained/service-plan-v15-route-choice/).

To reproduce conversion from the locally acquired PBF:

```sh
cd ../editoast
cargo build --release -p osm_to_railjson --features geos/static
./target/release/osm_to_railjson \
  ../ops-intelligence/data/rail/osm-belgium/belgium-261015.osm.pbf \
  ../ops-intelligence/data/rail/osm-belgium/belgium-261015.railjson \
  --generate-signals
cd ../ops-intelligence
uv run osrd-osm-railjson-normalize \
  data/rail/osm-belgium/belgium-261015.railjson \
  data/rail/osm-belgium/belgium-261015-deduplicated.railjson
```

The new fixture imports used by these cohorts are reproducible from OSRD's checked-in
RailJSON and rolling-stock examples. Use the infrastructure IDs returned by Editoast
in the manifests (the values below are from the collection run):

```sh
curl -X POST -H 'x-osrd-skip-authz: true' -H 'Content-Type: application/json' \
  --data-binary @tests/data/infras/etcs_infra/infra.json \
  'http://localhost:8090/infra/railjson?name=ops-intelligence-etcs&generate_data=true'
curl -X POST -H 'x-osrd-skip-authz: true' -H 'Content-Type: application/json' \
  --data-binary @tests/data/rolling_stocks/etcs_level2_rolling_stock.json \
  'http://localhost:8090/rolling_stock/'
curl -X POST -H 'x-osrd-skip-authz: true' -H 'Content-Type: application/json' \
  --data-binary @tests/data/infras/overlapping_routes/infra.json \
  'http://localhost:8090/infra/railjson?name=ops-intelligence-overlap-infra&generate_data=true'
curl -X POST -H 'x-osrd-skip-authz: true' -H 'Content-Type: application/json' \
  --data-binary @tests/data/infras/nested_switches/infra.json \
  'http://localhost:8090/infra/railjson?name=ops-intelligence-nested-switches&generate_data=true'
```

For exact route, seed, headway, and split provenance, use each checked-in cohort's
`manifest.json`. The training report also records every source manifest and episode
checksum.

To regenerate a cohort on the medium fixture, provide several background routes.
Repeat with the seeds, parameters, and scenario groups recorded in the checked-in
dataset manifests for each cohort:

Import the Y-shaped fixture too when regenerating its branch cohorts:

```sh
curl -X POST -H 'x-osrd-skip-authz: true' -H 'Content-Type: application/json' \
  --data-binary @tests/data/infras/y_infra/infra.json \
  'http://localhost:8090/infra/railjson?name=ops-intelligence-y-infra&generate_data=true'
```

```sh
uv run osrd-replay-collect --editoast-url http://localhost:8090 --infra-id 2 \
  --rolling-stock fast_rolling_stock --start-track TI0 --start-offset 100 \
  --end-track TH1 --end-offset 4900 \
  --background-route TE1:100:TF1:6400 \
  --background-route TB0:100:TH1:4900 \
  --background-route TI0:100:TF1:6400 \
  --seed 20261025 --episodes 80 --trains 8 \
  --headway-options-s 180 300 420 --max-departure-shift-s 300 \
  --max-dwell-s 180 --intervention-type service-insertion \
  --scenario-group medium-mixed-corridors-insertion-v1 \
  --output data/replays/medium-mixed-corridors-insertion-v1
```

Reproduce the earlier v3 three-group model with:

```sh
uv run osrd-replay-gnn-train \
  datasets/replay-insertion-v2/train-medium-northwest \
  datasets/replay-insertion-v2/train-medium-north-south \
  datasets/replay-insertion-v2/train-y-branch-a \
  datasets/replay-insertion-v2/validation-medium-northwest \
  datasets/replay-insertion-v2/validation-y-branch-a \
  datasets/replay-insertion-v2/test-medium-southeast \
  datasets/replay-insertion-v2/test-small-infra \
  datasets/replay-insertion-v2/test-y-branch-b \
  --validation-scenario-groups medium-mixed-corridors-insertion-northwest-v2 \
    y-infra-mixed-branch-insertion-a-validation-v1 \
  --test-scenario-groups medium-mixed-corridors-insertion-southeast-v1 \
    small-infra-mixed-corridor-insertion-v1 y-infra-mixed-branch-insertion-b-v1 \
  --epochs 300 --patience 30 --seed 61 \
  --output-dir models/trained/service-insertion-v3
```

The expanded v5 transfer evaluation uses the additional four train/test cohorts:

```sh
uv run osrd-replay-gnn-train \
  datasets/replay-insertion-v2/train-medium-northwest \
  datasets/replay-insertion-v2/train-medium-north-south \
  datasets/replay-insertion-v2/train-y-branch-a \
  datasets/replay-insertion-v3/train-etcs-west-southeast \
  datasets/replay-insertion-v3/train-etcs-west-northeast \
  datasets/replay-insertion-v4/train-etcs-central-north \
  datasets/replay-insertion-v4/train-etcs-high-density \
  datasets/replay-insertion-v2/validation-medium-northwest \
  datasets/replay-insertion-v2/validation-y-branch-a \
  datasets/replay-insertion-v2/test-medium-southeast \
  datasets/replay-insertion-v2/test-small-infra \
  datasets/replay-insertion-v2/test-y-branch-b \
  datasets/replay-insertion-v3/test-overlapping-routes \
  datasets/replay-insertion-v4/test-nested-switches \
  --validation-scenario-groups medium-mixed-corridors-insertion-northwest-v2 \
    y-infra-mixed-branch-insertion-a-validation-v1 \
  --test-scenario-groups medium-mixed-corridors-insertion-southeast-v1 \
    small-infra-mixed-corridor-insertion-v1 y-infra-mixed-branch-insertion-b-v1 \
    overlapping-routes-heldout-transfer-v1 nested-switches-heldout-transfer-v1 \
  --epochs 300 --patience 30 --seed 61 \
  --output-dir models/trained/service-insertion-v5
```

This model estimates which existing services become newly conflict-involved
after a proposed service is inserted; it does not synthesize a complete timetable
or recommend compensating retimings. A separate retiming artifact trained on 160
medium-fixture episodes and tested on 80 small-infra episodes achieved AP 0.176
versus 0.161 for linear, a 0.015 gain below the 0.05 offline gate; its arrival
head also lost to zero-change. Its datasets and rejected artifact are preserved
under `datasets/replay-retime-v1/` and `models/trained/service-retime-v1/` for
reproduction. OSRD simulation remains the feasibility authority.

Reproduce the collection after importing the fixture into a disposable OSRD
development database:

```sh
curl -X POST -H 'x-osrd-skip-authz: true' -H 'Content-Type: application/json' \
  --data-binary @tests/data/infras/medium_infra/infra.json \
  'http://localhost:8090/infra/railjson?name=ops-intelligence-medium&generate_data=true'
cd ops-intelligence
uv run osrd-replay-collect --editoast-url http://localhost:8090 --infra-id 2 \
  --rolling-stock fast_rolling_stock --start-track TI0 --start-offset 100 \
  --end-track TH1 --end-offset 4900 --seed 20261017 --episodes 60 --trains 8 \
  --headway-options-s 180 300 420 --max-departure-shift-s 300 --max-dwell-s 180 \
  --intervention-type service-insertion --scenario-group medium-northwest-southeast \
  --output data/replays/medium-northwest-southeast
uv run osrd-replay-gnn-train data/replays/service-insertion-dense \
  data/replays/service-insertion-medium data/replays/service-insertion-sparse \
  data/replays/medium-northwest-southeast \
  --validation-scenario-groups service-insertion-medium \
  --test-scenario-groups medium-northwest-southeast --epochs 300 --patience 25 \
  --seed 29 --output-dir data/replays/medium-infra-holdout-gated
```

The previous two-way dense-group GNN result is historical; use all three
headway groups with explicit validation and test partitions for the current
training regime:

```sh
uv run osrd-replay-gnn-train \
  data/replays/service-insertion-dense \
  data/replays/service-insertion-medium \
  data/replays/service-insertion-sparse \
  --validation-scenario-groups service-insertion-medium \
  --test-scenario-groups service-insertion-dense \
  --output-dir data/replays/service-insertion-gnn-full-regime
uv run osrd-replay-train \
  data/replays/service-insertion-dense data/replays/service-insertion-medium \
  --test-fraction 0.25 --output-dir data/replays/service-insertion-linear-test-dense
```

## Evaluate the overlap baseline

The first transparent conflict-risk reference predicts that a train is conflict
involved when its nearest departure is closer than a learned threshold. The
threshold is tuned only on training episodes; a stable hash split holds out whole
episodes. Pass separate source runs so sparse and dense scenarios remain
traceable:

```sh
uv run osrd-overlap-baseline \
  data/replays/run-001 data/replays/dense \
  --test-fraction 0.2 \
  --output data/replays/overlap-baseline-report.json
```

This is a physics-style benchmark, not a fitted production model. It reports a
majority-class reference, test precision/recall/F1, paired arrival-delta regression
references, and its exact split. When `scenario_group` is present, the evaluator
holds out entire scenario groups; at least two groups are required. Older replay
episodes without groups fall back to episode-level assignment.

The current local synthetic benchmark contains 16 episodes and 96 schedules over
two scenario families. Holding out `family-b` yielded departure-gap test precision
0.872, recall 0.971, and F1 0.919 (48 schedules, 35 conflict-involved). The
zero-change arrival-delta reference had 27.3 s MAE on the held-out schedules; the
training-mean reference had 45.7 s MAE. This is only a small, generated benchmark:
do not interpret it as field accuracy, calibrated risk, or evidence of operational
readiness. Local generated data and reports are ignored by Git.

## Train the first non-LLM replay models

`osrd-replay-train` derives the fixed `osrd-ops-replay-features/v2` vector from
paired v2 episodes and trains a regularized logistic conflict classifier plus a
regularized linear arrival-delta regressor. It reuses the scenario-group split,
fits scaling and weights only on training groups, tunes the classifier threshold
on training predictions for **newly introduced conflict involvement**, and writes
`model.json` and `report.json`:

```sh
uv run osrd-replay-train \
  data/replays/family-a data/replays/family-b \
  --test-fraction 0.2 --output-dir data/replays/trained
```

Earlier v1 runs used *all* treatment conflict involvement as their classification
target; their reported classifier metrics are not comparable with the current
new-conflict target. The arrival-delta regressor results remain paired simulation
comparisons. These are single-split synthetic benchmarks, not operational
accuracy claims. Artifacts record the feature schema, source checksums, and
training groups for review and reproduction.

An expanded run adds the `wide-headway` and `dense-headway` groups and holds out the
entire dense group. On its 30 schedules, the arrival-delta regressor reached 13.0 s MAE
versus 26.5 s for zero-change. The held-out dense group has conflicts for all 30
schedules, so its conflict-classification score is not informative. This is still a
single small infrastructure and a single held-out scenario group; the data model learns
incident-delay effects, not the impact of adding a requested service or applying
recovery edits. Reproduce it with:

```sh
uv run osrd-replay-train \
  data/replays/v2-family-a data/replays/v2-family-b \
  data/replays/v2-wide data/replays/v2-dense \
  --test-fraction 0.25 --output-dir data/replays/v2-four-family-trained
```

### Train the replay impact GNN prototype

`osrd-replay-gnn-train` represents each train as a node and connects consecutive
services by treatment departure order. A two-layer GATv2 model estimates each
train's paired arrival-time change and whether it became newly conflict-involved
in treatment relative to baseline. It uses the same whole-scenario-group split as the linear benchmark and
stores a PyTorch checkpoint plus a JSON report:

### Current coupled service-plan model (v34)

The current multi-network research artifact is
[`models/trained/service-plan-v34-multi-network-pristine-holdout/`](models/trained/service-plan-v34-multi-network-pristine-holdout/).
It predicts per-service conflict involvement and arrival change for prebuilt
candidate episodes. The complete model card, checksummed source manifests,
training history, group split, final evaluation, and validation-selected
action-cost weight are in that directory.

The training corpus is 31,920 paired OSRD episodes. Train and validation data
come from Swiss and Belgian cohorts under
`datasets/replay-service-plan-v7/{swiss-coupled-retiming-v1,belgium-osm-coupled-retiming-v2}`;
the final test uses four newly collected sites in each network under
`datasets/replay-service-plan-v7/{switzerland-osm-coupled-retiming-pristine-holdout-v1,belgium-osm-coupled-retiming-pristine-holdout-v1}`.
The v34 `report.json` records exact cohort paths, checksums, and scenario-group
membership. Swiss Actual and Infrabel delay profiles condition the simulated
episodes; conflict and delay impact labels are still OSRD simulation outputs.

Recreate the v34 fit with its fixed group partitions and pristine test roots:

```sh
uv run osrd-replay-gnn-train-corpus \
  --train-root datasets/replay-service-plan-v7/swiss-coupled-retiming-v1/train \
  --train-root datasets/replay-service-plan-v7/belgium-osm-coupled-retiming-v2/train \
  --validation-root datasets/replay-service-plan-v7/swiss-coupled-retiming-v1/validation \
  --validation-root datasets/replay-service-plan-v7/belgium-osm-coupled-retiming-v2/validation \
  --test-root datasets/replay-service-plan-v7/switzerland-osm-coupled-retiming-pristine-holdout-v1/test \
  --test-root datasets/replay-service-plan-v7/belgium-osm-coupled-retiming-pristine-holdout-v1/test \
  --output-dir models/trained/service-plan-v34-reproduction \
  --seed 17 --epochs 300 --patience 25 --batch-size 8 --device cpu --log-every 1
```

The command verifies each cohort's episode SHA-256, derives scenario groups from
the cohort contents, rejects duplicate/cross-partition groups, and delegates to
the grouped trainer. Its output report records the resolved sources and split.

Example prediction for a candidate episode containing the proposed schedule and
the baseline OSRD output, but no treatment outcomes or labels:

```sh
uv run osrd-replay-gnn-predict \
  --model models/trained/service-plan-v34-multi-network-pristine-holdout/model.pt \
  --episode candidate-episode.json \
  --output candidate-impact.json
```

This model fails the pre-set worst-group ranking gate, so it is not eligible for
the planner-ranking endpoint. Its outputs are research estimates; run every
complete candidate through OSRD before treating it as feasible. The recorded
validation-call comparisons are retrospective, not measured runtime savings.

### Direct plan-level graph scorer (research)

`osrd-replay-plan-gnn-train` is a separate experiment that pools the full
action-conditioned service graph and predicts plan-level conflict count and
positive existing-service arrival delay. It learns pairwise ordering among
alternatives sharing the same scenario, seed, and episode index. Conflict count
has priority over delay in its explicit training utility. This removes the
hand-tuned post-hoc blend used by the v34 evaluator; it does not learn operator
preferences or remove OSRD validation.

Use disjoint scenario-group cohort roots. The v34 test roots shown below have
already been inspected and therefore support diagnostic comparisons only; do
not treat a run against them as a new final test. For model development, fit on
the training roots and select by validation. Before reporting a final
generalization result, collect a fresh set of spatially held-out sites.

```sh
uv run --extra gnn osrd-replay-plan-gnn-train \
  --train-root datasets/replay-service-plan-v7/swiss-coupled-retiming-v1/train \
  --train-root datasets/replay-service-plan-v7/belgium-osm-coupled-retiming-v2/train \
  --validation-root datasets/replay-service-plan-v7/swiss-coupled-retiming-v1/validation \
  --validation-root datasets/replay-service-plan-v7/belgium-osm-coupled-retiming-v2/validation \
  --test-root datasets/replay-service-plan-v7/switzerland-osm-coupled-retiming-pristine-holdout-v1/test \
  --test-root datasets/replay-service-plan-v7/belgium-osm-coupled-retiming-pristine-holdout-v1/test \
  --output-dir models/trained/plan-gnn-v1 --epochs 150 --patience 20
```

The trainer validates cohort checksums and disjoint scenario groups, fits
feature normalization on training episodes only, hashes the JSONL data as it
converts each episode to a compact graph, checkpoints after each epoch, and
restores the checkpoint with the lowest macro-average validation loss across
scenario groups. This gives each held-out group equal weight in model selection;
pooled validation loss remains in the report for comparison. The report also
includes macro-average and worst-group validation/test loss. Resume a stopped
process with the same command plus `--resume`. Test metrics are labeled diagnostic-only
because this corpus's test results have been seen in earlier research. The
model artifact is advisory; all alternatives must pass complete OSRD validation.

#### Fresh spatial holdout result (v3)

The v3 experiment uses the corrected-millimetre v10 replay cohorts. Training,
validation, and final test sites are disjoint across Belgium and Switzerland;
the final test covers eight previously unused sites (1,920 episodes, 160
contexts). Training also includes four older corrected cohorts for stop-choice
coverage, all confined to training. Checksum and group-disjointness audits passed.
Checkpoint selection used equal-weight scenario-group macro validation loss.

On the untouched test, the graph scorer reached 0.967 pooled pairwise
concordance (0.967 macro by site), 0.881 minimum-conflict top-choice rate, and
0.169 mean top-choice conflict regret. The fixed train-only Ridge reference
scored 0.911 concordance, 0.688 minimum-conflict rate, and 0.356 regret; the
minimum-edit rule scored 0.291, 0.056, and 1.731. The graph scorer exceeded
Ridge concordance at all eight sites; its worst site scored 0.945 versus
0.808 for Ridge. Exact site-level metrics and episode hashes are preserved in
`models/trained/plan-gnn-v3-v10-spatial-macro/generalization-baselines.json`.

This supports generalization across these synthetic OSRD sites and scenarios,
not to operator choices or real-world operations. The existing service-plan
promotion gate is based on per-group conflict average-precision improvement;
this plan-level experiment predicts multi-objective simulated outcomes and does
not establish that gate. Keep learned ranking disabled. Any later promotion
also requires an operationally grounded, per-group benchmark using reviewed
timetable alternatives and operator choices alongside OSRD feasibility and
outcomes, with acceptance criteria frozen before evaluation.

#### Third-network transfer gate: Île-de-France (v4)

The v3 checkpoint was frozen by SHA-256 before opening the France replay holdout.
We imported a dated Geofabrik Île-de-France OSM extract as research infrastructure,
then collected 2,240 corrected-millimetre OSRD candidate episodes across four
preselected sites. Each site has 40 matched contexts (20 under each borrowed Belgian
and Swiss delay-profile stratum); the French profiles are not observed French delay
evidence. Episode checksums passed, all four sites have more than 1,000 comparable
candidate pairs, and scenario groups and episode hashes are disjoint from training
and validation. Source details, OSM attribution, converter warnings, identity
normalization, and limits are recorded in
`data/rail/osm-france/source-manifest.json`.

The frozen v3 graph model scored 0.962 pooled pairwise concordance and 0.957 macro
concordance across sites, versus 0.869 pooled and 0.848 macro for train-only Ridge.
It selected a minimum-conflict alternative in 86.3% of contexts with 0.200 mean
conflict regret; Ridge reached 53.1% and 0.606. However, the precommitted per-site
gate **failed**. At site 1096502398, GNN concordance was 0.938 versus Ridge 0.936
(below the required +0.05), its minimum-conflict selection was 80.0% versus 90.0%,
and regret was 0.350 versus 0.175. At site 2635782906, its +0.049 concordance
margin also fell just short of +0.05, despite better selection and regret. The
other two sites passed all three metric conditions. Full metrics, hashes, and
the frozen gate result are preserved in
`models/trained/plan-gnn-v4-france-transfer/`.

The France result does not justify promoting learned ranking. Keep it disabled.
The v4 evaluation remains a one-shot result; France is now seen development data
and cannot serve as another independent holdout. This benchmark only measures
transfer across OSM-derived OSRD simulations; labels are synthetic, the topology
is not authoritative, and OSRD validation remains the final authority. The
follow-up v5 candidate added an explicit action summary and a validation-selected
40% GNN / 60% Ridge rank blend, then faced a frozen multi-site Netherlands holdout.
The Netherlands gate **failed**: the blend's concordance uplift over train-only
Ridge ranged from -0.001 to +0.008 across the four sites, below the required +0.05
at every site. At Vlissingen Souburg it also had worse mean conflict regret (0.150
vs 0.100) and a lower minimum-conflict selection rate (90.0% vs 92.5%). Macro
concordance was 0.953 vs 0.950 for Ridge, but pooled performance cannot override
the site-level gate. Integrity checks passed, with 40 contexts and at least 1,635
comparable pairs per site. The one-shot result and detailed metrics are recorded in
`models/trained/plan-gnn-v5-netherlands-transfer/GATE_RESULT.md`; do not retune or
rerun on these outcomes. Keep learned ranking disabled. Netherlands labels are
OSRD simulations conditioned on borrowed Belgian and Swiss delay profiles over
OSM-derived topology, not Dutch operator outcomes. Any later operational
promotion also requires reviewed planner choices and a new untouched holdout.

To rank candidate episodes, pass a JSON array of outcome-free episodes (or an
object with an `episodes` array). Each episode must include the proposed
`inputs.schedule_payloads`, `inputs.baseline_schedule_payloads`, baseline
simulation summaries, and pre-simulation resource paths. Treatment labels are
not consumed:

```sh
uv run --extra gnn osrd-replay-plan-gnn-predict \
  --model models/trained/plan-gnn-v1/model.pt \
  --candidates candidate-episodes.json \
  --output ranked-candidates.json
```

The returned score is lower-is-better and orders only the supplied alternatives.
The model does not generate routes or schedules, and the score is not calibrated
as a probability. Validate the complete top-ranked timetable in OSRD and retain
the planner's configured policy constraints.

```sh
uv run osrd-replay-gnn-train \
  data/replays/v2-family-a data/replays/v2-family-b \
  data/replays/v2-wide data/replays/v2-dense \
  --test-fraction 0.25 --output-dir data/replays/v2-four-family-gnn
```

For a strict corridor holdout, pass `--test-scenario-groups` to the GNN,
linear, or overlap-baseline trainer. The full GNN regime also accepts
`--validation-scenario-groups`; both partitions exclude whole groups. The final
test groups are not used for early stopping or threshold selection. For example:

```sh
uv run osrd-replay-gnn-train \
  data/replays/service-insertion-dense \
  data/replays/service-insertion-medium \
  data/replays/service-insertion-sparse \
  data/replays/service-insertion-north-south \
  --validation-scenario-groups service-insertion-medium \
  --test-scenario-groups service-insertion-north-south \
  --epochs 300 --patience 25 --seed 17 \
  --output-dir data/replays/route-holdout-west-to-north-south
```

The trainer validates source manifests and checksums, fits feature scaling on
training groups only, tunes its conflict threshold against validation predictions,
reduces the learning rate on validation plateaus, and restores the best validation
checkpoint after early stopping. It writes `model.pt`, `report.json`,
`training-history.jsonl`, and `MODEL_CARD.md`; `checkpoint-last.pt` is atomically
updated after each epoch. Resume an interrupted run with the same data and settings
plus `--resume`. The checkpoint signature rejects changed source data, groups, seed,
features, architecture, or optimizer settings. Use a new output directory or
`--overwrite` to start a fresh run.

Score an unrun candidate by creating an episode JSON with its proposed
`inputs.schedule_payloads`, the baseline schedule payloads, and the baseline arm
under `outputs.paired.baseline` (schedule IDs and simulation summaries). Treatment
outputs and `labels` are not required. The predictor emits per-service conflict
scores. It returns arrival-delta values only when that head passes its own gate.
For v34 that gate passes on the paired OSRD benchmark, but those estimates are
not calibrated against real movement outcomes:

```sh
uv run osrd-replay-gnn-predict \
  --model data/replays/route-holdout-west-to-north-south/model.pt \
  --episode candidate-episode.json \
  --output candidate-impact.json
```

The output is explicitly advisory and requires full OSRD validation. Historical
insertion-only artifacts have higher aggregate conflict AP than the linear reference,
but it fails the minimum per-test-group margin; the predictor suppresses threshold
decisions. The earlier v3 artifact passed that criterion on a narrower synthetic
evaluation. Neither is production-ready. The planner still needs real-network
simulation episodes and a measured end-to-end candidate-generation benefit. The GNN
scores a supplied complete candidate timetable; it does not create route, stop,
frequency, or timetable edits.

The plan-level scorer has newer research artifacts at
[`models/trained/plan-gnn-v3-route-aware/`](models/trained/plan-gnn-v3-route-aware/)
and earlier baselines at [`models/trained/plan-gnn-v1/`](models/trained/plan-gnn-v1/).
V3 adds a small station-stop/dwell grid (512 stop episodes, eight operational
points, two nonzero dwell patterns) to a 300/600-second frequency and 0/60-second
phase grid. Its validation pairwise concordance is 0.941 and diagnostic test
concordance is 0.977 across only 24 contexts and three corridors in two OSM-derived
networks. The corpus has no route-waypoint actions or retiming, and the delay head
has high aggregate error. This simulation-trained research ranker has no real
alternative-timetable outcomes or operator choices, does not create routes or
schedules, and is not wired into the planner API. It only orders supplied replay
graphs; OSRD remains authoritative for feasibility. See the model card and
[readiness report](../docs/ai/training-data-and-model-readiness.md) for scope and
safeguards.

This scorer cannot currently be plugged into the live frequency planner. The
insertion-ranking contract accepts exactly one new schedule per candidate, while
the service planner expands a frequency request into 1–50 departures and can
also vary route, first-departure phase, and eligible-service retimings. Those
repeated-service options are a different model input and evaluation problem;
do not apply single-insertion scores to them as if they were validated rankings.
A future planner ranker needs paired OSRD examples for complete repeated-service
options and an end-to-end comparison against the current deterministic ordering.

The graph scorer's input reader checks checksums incrementally and retains graph
tensors rather than parsed JSON episodes. It checkpoints every epoch and supports
`--resume`, so long full-corpus fits can continue after a terminal interruption.

This is an architecture experiment on a tiny synthetic dataset from one OSRD
infrastructure. Its conflict labels measure newly introduced OSRD conflict
involvement, and its delay labels describe the tested departure
perturbations. The later service-insertion slice below covers one added-service
scenario, but the model does not choose its route, stops, or departure time. Treat
held-out metrics as exploratory only; OSRD simulation and conflict validation
remain authoritative.

The initial run incorrectly used all treatment conflicts as its classification
target; that run is superseded. With the corrected newly introduced conflict
target, the held-out dense group has zero positive labels, so its classification
metric is not evidence of detection. Arrival-delta MAE is 26.62 s, slightly worse
than zero-change at 26.50 s and substantially worse than the linear model's 13.04 s
on the same split. That retiming-only GNN run has not demonstrated predictive
value. The later service-insertion result below evaluates a more relevant action;
multi-route and repeated held-out-group evaluation remain before changing model
capacity.

## Run beside the OSRD Compose stack

From the repository root, build and start only this service:

```sh
docker compose -f docker-compose.yml -f ops-intelligence/docker-compose.yml \
  up -d --build ops-intelligence
```

For Podman Compose, replace `docker compose` with `podman compose`.

## Fetch the external Belgian delay benchmark

The RIDE Gold Lite release is substantially larger than its compact shared
evaluation table. Fetch only that pinned core artifact first; it is about 74 MB
and the command verifies its size and SHA-256 before writing a provenance
manifest. Data and manifests land under the ignored `data/` directory and are
not committed:

```sh
uv run osrd-ride-fetch-core
```

The pinned table is from RIDE revision
`e5c1ed1126bca1784eb6ea93816dd6cc89afa544` and is licensed CC BY 4.0. Retain the
generated attribution to RIDE, Infrabel, and Open-Meteo if data-derived results
are shared. Install `uv sync --extra rail-data` to inspect Parquet locally. This
external observed-delay benchmark is a separate track from synthetic OSRD
paired replay labels: it cannot label conflict causation or timetable-change
counterfactuals.

With the optional reader installed, score the no-learning persistence reference
against every available future-delay target (1–15 events). The script reports
overall and per-horizon MAE, RMSE, signed bias, and valid-label counts. It reads
only the fixed RIDE test table and fits no model:

```sh
uv sync --extra rail-data
uv run osrd-ride-baseline
```

On the pinned evaluation table the persistence reference scored 97.5 s MAE,
236.9 s RMSE, and -35.3 s mean bias over 6,528,133 valid horizon targets. MAE
rose from 56.0 s at event 1 to 133.2 s at event 15. The full horizon report is
written to ignored local data; keep the fixed test split out of model fitting and
hyperparameter selection.

## Initial boundaries

- Planning request models require timezone-aware timestamps and explicit snapshot
  bindings (`infra_version`, `timetable_revision`, `osrd_version`).
- Unknown request fields are rejected to make contract changes explicit.
- `EditoastOSRDClient` reads and binds the source timetable, validates complete
  isolated candidates against OSRD simulation and conflict APIs, and removes its
  temporary timetable and schedule set. It does not apply a recommendation.
- `optimizer.departure_shifts` now generates deterministic, bounded departure
  shift edits for explicitly affected schedules, excluding protected services.
  Candidate IDs bind the complete infrastructure, timetable, and OSRD version
  snapshot. It returns deltas, not a replacement timetable or a claim that
  service will improve. Only candidates that pass the complete-timetable OSRD
  simulation and conflict checks are returned as accepted.
- No model artifacts, operational data, database connection, or OSRD credentials
  are included in this scaffold.
