# OSRD Operations Intelligence

Initial scaffold for the advisory planning service described in
[`docs/ai/ai-native-network-operations-design.md`](../docs/ai/ai-native-network-operations-design.md).

This service is deliberately isolated from OSRD core. Its planning API generates
advisory candidates, validates them through OSRD, and never applies them to an
operational timetable. The first implementation slice was an offline replay collector: it creates isolated synthetic timetables in
Editoast, records simulation summaries and conflict outputs, and writes seeded,
version-bound JSONL episodes. It never edits a pre-existing operational timetable.

## Layout

```text
src/osrd_ops_intelligence/
  api/         internal request/response contracts (pydantic)
  service/     FastAPI app and OSRD client boundary
  features/    versioned feature transforms (placeholder)
  models/      linear and quantile-MLP baselines; graph model planned
  optimizer/   candidate generation, objectives (placeholder)
  training/    replay collector; labels, splits, training (later)
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

OpenAPI is available at `http://localhost:8095/docs`; health is at
`http://localhost:8095/healthz`.

## Validate bounded timetable candidates

The planning route supports delay incidents with an explicit delay and affected
train-schedule IDs. It creates bounded departure-shift candidates, clones the
complete regular-schedule set into a temporary unpublished timetable, runs
OSRD simulation summaries and conflict detection, deletes the temporary records,
and returns accepted and rejected candidates. It never edits or publishes the
source timetable. Obtain the current exact binding first with
`GET /v1/operations/snapshots/{infra_id}/{timetable_id}`; its timetable revision
is a SHA-256 over the normalized OSRD train-schedule forms.

Configure `OSRD_EDITOAST_URL` (for the repository Compose network,
`http://gateway:4000/api`) and, where needed, `OSRD_AUTHORIZATION`. The adapter
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
RIDE Gold Lite provides a pinned Belgian train/test forecasting dataset; Swiss Actual
data and GTFS are assessed as later cross-network sources. Neither replaces OSRD
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
terminal progress reporting, and resumable Xet cache state. If Xet reports a known CAS
response-decoding/reconstruction failure, it falls back to a resumable HTTPS range
transfer with bounded retries. A Hugging Face token is not required for this public
dataset, but is recommended for a multi-gigabyte fetch: the Hub warns that anonymous
requests have lower limits and may be slower. A configured `HF_TOKEN` or saved login
is used automatically. SHA-256 hashes are verified before data is accepted. Any `.part`
file left by the older single-stream downloader is not reused by Xet and is removed
only after the replacement verifies successfully.

To fit the bounded CPU baseline after acquisition, install the numeric extra and run:

```sh
uv sync --extra rail-data
uv run osrd-ride-train --output-dir data/rail/ride-forecast-model
```

The trainer verifies every pinned input, memory-maps the 5.7 GB feature matrix, samples
up to 20,000 fit rows and 5,000 later rows grouped by service date. Dates after the
default 2024-10-01 cutoff are split chronologically into early-stop, calibration, and
internal-evaluation groups. It optimizes only valid targets and does not load or score
the 2025 holdout. Its report compares each horizon with a train-only mean, includes
signed bias, and breaks evaluation metrics out by service month. Metrics are in
provider-normalized target units; see the readiness plan for interpretation and limits.

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

RIDE Gold Lite also publishes a PyTorch Geometric heterogeneous GNN representation.
The schemas and checksums we retrieved are recorded in
[`data_contracts/ride/gold-lite-gnn-manifest.json`](data_contracts/ride/gold-lite-gnn-manifest.json),
and the graph/data design and acquisition constraints are in the readiness document.
The GNN shards are not downloaded by the tabular fetch command. From the upstream RIDE
repository root, `python -m scripts.dataset.download.download_dataset --target
gold_lite_gnn` is the official route, but fetches the whole GNN component including its
test shard. For controlled training, selectively fetch only `gnn/feature_spec.yaml`,
`gnn/normalization.yaml`, and `gnn/train/*.pt` from revision
`e5c1ed1126bca1784eb6ea93816dd6cc89afa544`, then verify the manifest hashes before
loading. The first graph model should be an optional offline PyTorch/PyG path that
predicts the same 15 delay horizons and masks as the tabular models, with date-grouped
splits and shared event-level scoring. Do not use the test shard until model selection
is frozen. The published RIDE graph represents train/station/event interactions; it is
not an OSRD track/switch/signal graph and cannot replace OSRD validation of a timetable.

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
pinned infrastructure and rolling-stock versions. Treatment includes one seeded
synthetic departure delay. Labels include conflict involvement, newly involved
trains, simulation status, and matched arrival, travel-time, and energy deltas.
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

`osrd-replay-train` derives the fixed `osrd-ops-replay-features/v1` vector from
paired v2 episodes and trains a regularized logistic conflict classifier plus a
regularized linear arrival-delta regressor. It reuses the scenario-group split,
fits scaling and weights only on training groups, tunes the classifier threshold
on training predictions, and writes `model.json` and `report.json`:

```sh
uv run osrd-replay-train \
  data/replays/family-a data/replays/family-b \
  --test-fraction 0.2 --output-dir data/replays/trained
```

On the current local data, the classifier reached F1 0.854 on the held-out family
(precision 0.745, recall 1.0), below the departure-gap reference's F1 0.919. The
arrival-delta regressor reached 9.9 s MAE, compared with 27.3 s for zero-change and
45.7 s for the training-mean reference. These are single-split results from only
two synthetic scenario groups; the trained weights are research artifacts, not
operational models. The artifact records the feature schema, source checksums,
and training groups for review and reproduction.

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
