# OSRD Operations Intelligence

Initial scaffold for the advisory planning service described in
[`docs/ai/ai-native-network-operations-design.md`](../docs/ai/ai-native-network-operations-design.md).

This service is deliberately isolated from OSRD core. The current implementation
provides a typed API shell and health endpoint only; it does not forecast delays,
generate candidate plans, call Editoast, or apply timetable changes. The plan
endpoint returns `501 planning_not_configured` until the orchestration and OSRD
adapter are implemented.

## Local development

Requires Python 3.11+ and `uv`.

```sh
uv sync --extra dev
uv run uvicorn osrd_ops_intelligence.api:app --app-dir src --reload --port 8095
```

OpenAPI is available at `http://localhost:8095/docs`; health is at
`http://localhost:8095/healthz`.

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
