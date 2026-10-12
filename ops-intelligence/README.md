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
every episode. `manifest.json` captures run parameters and content hash;
`episodes.jsonl` records exact schedule inputs, Editoast simulation outputs,
conflicts, and derived labels. `simulation_by_train_schedule_id` preserves status,
travel time, energy and path-item times; `conflict_count` and `has_conflicts` are
labels. JSONL is the initial portable interchange format; promotion to Parquet is
deferred until feature columns and stable nullability are agreed.

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
majority-class reference, test precision/recall/F1, and its exact episode split.

## Run beside the OSRD Compose stack

From the repository root, build and start only this service:

```sh
docker compose -f docker-compose.yml -f ops-intelligence/docker-compose.yml \
  up -d --build ops-intelligence
```

For Podman Compose, replace `docker compose` with `podman compose`.

## Initial boundaries

- Planning request models require timezone-aware timestamps and explicit snapshot
  bindings (`infra_version`, `timetable_revision`, `osrd_version`).
- Unknown request fields are rejected to make contract changes explicit.
- `OSRDClient` is a protocol, not a live client. A future implementation must
  validate the complete candidate timetable through OSRD and preserve the exact
  snapshot binding.
- No model artifacts, operational data, database connection, or OSRD credentials
  are included in this scaffold.
