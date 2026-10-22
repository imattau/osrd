# AI goal: use time-dependent station patronage in OSRD planning

## Objective

Help planners compare OSRD stopping schedules and timetables using aggregated station
patronage by time of day. For each feasible option, show which observed station demand
periods the proposed calls cover, how coverage changes between options, and the source,
mapping, and assumptions behind the comparison.

## Scope

- Accept aggregated boardings and alightings by station, weekday, and local-time bucket.
- Preserve dataset source, license, observation period, timezone, and station-to-OSRD
  operational-point mapping provenance.
- Derive candidate station call times from OSRD-validated schedules and compare each call
  with demand at the same mapped operational point and local time.
- Show demand coverage beside feasibility and operational trade-offs. Offer an explicit
  deterministic demand-aware option order after OSRD feasibility and conflict checks. Keep
  OSRD simulation and conflict checks as the feasibility authority.
- Evaluate usefulness on held-out infrastructure groups before enabling model-ranked
  recommendations. Deterministic coverage summaries remain inspectable.

## Acceptance criteria

1. A versioned input contract rejects missing provenance, invalid timezones, duplicate
   station identities, invalid time buckets, and negative or non-finite volumes.
2. Candidate comparisons report covered and total boardings/alightings, matched and
   unmatched station calls, the exact metric version, and `unknown` rather than zero where
   demand data or an identity mapping is unavailable.
3. Coverage is calculated from OSRD simulation-derived calls, is time-zone and weekday
   aware, and never substitutes a station-name guess for a reviewed identity mapping.
4. The planner can inspect the patronage source and mapping alongside each comparison;
   demand coverage is not described as ridership, passenger benefit, or an outcome.
5. Evaluation compares demand-aware stopping and timetable options against the same
   OSRD-feasible alternatives without demand ranking, using infrastructure-held-out data
   and planner review. Any learned ranker remains research-only until a separate promotion
   gate is met.

## Initial implementation

The v1 JSON Schema, infrastructure-bound Python profile contract, and deterministic
station/time-bucket coverage function are in place. Single-service and complete calendar
service-plan requests can include a profile, and the planner UI can upload it. The OSRD
simulation summary provides final times for schedule path items; the service maps those
times to explicit endpoint and scheduled-stop calls and returns a coverage summary per
candidate, jointly across all proposed services for complete plans. Missing simulation
times and wholly unavailable operational-point mappings remain unknown. The optional
`station_patronage_coverage` order ranks options after feasibility and new-conflict count,
then before the selected retiming objective; candidates without comparable coverage rank
after candidates with available coverage. Complete-plan review records preserve this
deterministic ranking provenance, while complete-plan examples remain excluded from the
learned ranker's calendar/hourly training modes. Hourly-cycle plans can also opt into
deterministic coverage comparison by supplying operating weekdays and a local cycle start
minute in the profile timezone. OSRD path-item offsets are projected onto those local
weekday/minute slots, including calls after midnight; response records use weekday/minute
fields rather than implying a calendar service date. Hourly ranking remains feasibility-
and conflict-first, and the model remains research-only.

Complete calendar plan search also accepts `dwell_options_seconds` on an intermediate
waypoint. The bounded stop-pattern dimension compares required-stop dwell alternatives and,
for optional stops, passing versus each dwell choice. Candidate summaries return the chosen
dwell by waypoint, and the planner UI displays it. Planner-review feature schema v9 records
requested dwell-choice breadth and selected dwell magnitudes for future held-out evaluation;
this does not enable a learned ranker.

The next step is to evaluate the metric with real, licensed patronage profiles and
planner review, refine demand matching for service windows and multiple weekdays, and
extend weekday-aware stop design with route-supported station suggestions.

An offline preference-ranker path now captures separate boarding/alighting coverage,
availability and mapping/match-quality features. Its v3 model uses regularized quadratic
feature interactions so it can learn conditional trade-offs (for example, when a planner
accepts more retiming to cover a demand period). It supports a `--require-patronage` data
gate that requires 30 fully profile-backed choice sets in each infrastructure-held-out
split. It remains research-only: no licensed patronage review dataset has yet established
that these learned trade-offs generalize, and the live planner continues to expose the
deterministic coverage order only.

## Boundaries

Use only aggregated data with a documented license. Do not ingest passenger-identifiable
records. A covered station/time bucket is a service-availability proxy; it does not model
origin-destination journeys, transfers, crowding, capacity, accessibility, reliability,
or willingness to travel. Report these limitations wherever the metric is presented.
