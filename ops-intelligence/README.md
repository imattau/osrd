# OSRD Operations Intelligence

Initial scaffold for the advisory planning service described in
[`docs/ai/ai-native-network-operations-design.md`](../docs/ai/ai-native-network-operations-design.md).

This service is deliberately isolated from OSRD core. The planning API remains a
typed shell and does not generate or apply candidate plans. The first implementation
slice is an offline replay collector: it creates isolated synthetic timetables in
Editoast, records simulation summaries and conflict outputs, and writes seeded,
version-bound JSONL episodes. It never edits a pre-existing operational timetable.

## Layout

```text
src/osrd_ops_intelligence/
  api/         internal request/response contracts (pydantic)
  service/     FastAPI app and OSRD client boundary
  features/    versioned feature transforms (placeholder)
  models/      baselines and graph-temporal model (placeholder)
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
- `OSRDClient` is a protocol, not a live client. A future implementation must
  validate the complete candidate timetable through OSRD and preserve the exact
  snapshot binding.
- No model artifacts, operational data, database connection, or OSRD credentials
  are included in this scaffold.
