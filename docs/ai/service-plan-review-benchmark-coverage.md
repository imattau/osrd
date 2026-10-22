# OSRD planner review benchmark: coverage and collection plan

**Audit date:** 2026-10-21
**Status:** collection has not started; no preference model was fitted.

## Current evidence

The current workspace state was rechecked and the local Operations Intelligence audit
journal was exported with the review-data collector. The repository-local journal has
no review table yet; an existing workspace journal has a review table with zero rows.
The export contains zero submitted service-plan reviews, zero multi-option choices, and
zero records for either calendar or hourly timetables. No infrastructure groups are
represented. The preference dataset is empty and marked `training_ready: false`.
The audit inspected aggregate counts only; no planner review content was read or copied
into this report.

| Requirement | Current | Target | Gap |
|---|---:|---:|---:|
| Calendar multi-option choices | 0 | 240 | 240 |
| Hourly multi-option choices | 0 | 240 | 240 |
| Independent infrastructure groups | 0 | 8 | 8 |
| Choices per infrastructure, per mode | 0 | 30 | 30 at each group |

The thresholds imply 480 choice events total: 30 calendar and 30 hourly choice sets in
each of eight independently labeled rail infrastructures. No-feasible and no-suitable
records should also be captured to measure coverage, but they are not candidate-level
preference labels and do not count toward the 30 multi-option choices.

The latest aggregate export is available locally at
`ops-intelligence/data/service-plan-review-benchmark/manifest.json`; its event file is
empty (SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).
It reports `coverage_sufficient_for_preference_fit: false` and
`training_ready: false`.

## Capture and integrity changes

- New review records explicitly preserve the candidate display order. Exported event
  schema v4 carries this order forward; the trainer continues to read v3 exports, but
  only v4 blind-protocol choices can enter a training split.
- Hourly planner reviews now use the hourly request when deriving their request ID for
  journal persistence.
- Set `OSRD_OPS_BLIND_REVIEW_MODE=true` for a collection deployment. The service
  disables its configured insertion model, shows a blind-review banner, and rejects
  review submissions without the blind protocol marker.
- Also set `OSRD_OPS_REVIEW_INFRASTRUCTURE_GROUP` to a stable study-assigned label for
  the physical network. Blind collection fails to start without this value, and the API
  checks that submitted records match it. Do not use timetable IDs or separate database
  imports of one network as independent group labels.
- Preference fitting now accepts only explicitly blind, chosen-for-review events. It
  requires at least eight infrastructure groups and 30 such choices per group and mode.
  Unselected alternatives remain unlabeled; no-feasible and no-suitable events remain
  separate outcomes.
- The split-freezing command binds the infrastructure assignments to the full exported
  event-file SHA-256. The trainer requires this manifest and checks that no infrastructure
  crosses train, validation, and test partitions. The manifest also fixes top-1 as the
  primary metric, mean reciprocal rank and listwise loss as secondary metrics, and
  displayed-order plus uniform-random references before fitting.
- Split freezing and model fitting now also require the reviewed physical-infrastructure
  registry. It must map every study label to a unique canonical network identity and
  include observed source OSRD infrastructure IDs, an independence basis, reviewer role,
  and review timestamp. The registry digest is bound into the split manifest and research
  artifact; the automated check verifies consistency but does not independently establish
  that the attested networks are physically independent.

## Remaining collection plan

1. Deploy the planner in blind mode separately for each participating infrastructure,
   setting a stable `OSRD_OPS_REVIEW_INFRASTRUCTURE_GROUP` label and keeping deterministic
   display ordering active.
2. Ask planners to submit each review explicitly. Capture chosen-for-review, no-feasible,
   and no-suitable dispositions; optional categorical reasons remain planner-stated
   rationale, not outcome labels.
3. Export the review journal and inspect aggregate coverage by timetable mode and study
   infrastructure. Continue collection until every group has at least 30 multi-option
   choices in both modes.
4. Freeze separate calendar and hourly infrastructure splits from the final event-file
   digest. Review those assignments and metric definitions before fitting either model.
5. If coverage is sufficient, fit separate research-only listwise rankers and compare
   held-out top-1 accuracy, reciprocal rank, and listwise loss against displayed order
   and random choice, including per-infrastructure results. Preference fit will not
   establish operational acceptance or service outcomes, and the models remain out of
   production.

The current workspace contains no planner data that can satisfy step 1. Collection
requires planners to review real OSRD candidate sets on at least eight independently
labeled infrastructure groups. Synthetic replay outcomes cannot replace this evidence.

## Pilot access recheck

**Rechecked:** 2026-10-21

The local development stack is now running under rootless Podman, and its planner can
reach the local Editoast service. This resolves the container-runtime issue for local
integration work, but it does not provide the authenticated operator access or planner
participation required to start the prospective pilot.

- The active Operations Intelligence SQLite journal has zero `service_plan_reviews`
  rows. The exported preference event file remains empty and its manifest still reports
  `training_ready: false`. The local smoke request below was not a planner review and is
  not counted as preference data.
- A local advisory calendar planning request against synthetic infrastructure 2 and
  timetable 2 generated two options. OSRD pathfinding, simulation, and conflict checks
  accepted both; an explicitly synthetic patronage profile produced time-bucket coverage
  summaries. The request used no observed passenger data, and the source timetable
  revision was unchanged after validation. This verifies the container-to-OSRD planning
  integration and the deterministic coverage path only.
- The local seed database contains 17 infrastructure entries, primarily test fixtures
  and imported static examples. No evidence identifies participating planners or proves
  that any eight entries represent independent physical networks. A read-only scan of
  timetable-set IDs 1–1000 found no `HOURLY` timetable fixture, so this local stack cannot
  exercise the required hourly pilot mode. After that scan, local-only tests created
  synthetic scenario 12/timetable 153337 (hourly, initially empty) and scenario
  13/timetable 153341 (calendar, initially empty). The rebuilt planner accepted three
  hourly options and two complete-calendar options on these blank timetables; all passed
  OSRD simulation and conflict checks. The calendar request generated two departures and
  satisfied its simulated travel-time limit. Both requests used explicitly synthetic
  patronage profiles and were not saved as planner reviews. These fixtures demonstrate
  from-scratch integration, not operator participation or pilot evidence.
- A separate local demo account is configured with the mock gateway provider and the
  `Admin` role for local testing. It can see the 17 local fixture infrastructure entries,
  uses no password, and is confined to this machine-bound development stack; it is not a
  participating operator identity or pilot access.
- A live complete-calendar flow-on test used local timetable 153329 with six synthetic
  source services. The planner generated nine bounded candidates; eight passed OSRD
  validation, including options that retimed an eligible service. Selecting a retimed
  option in the planner created draft scenario 14/timetable 153362 with the new service
  and the proposed existing-service change. The source timetable remained at six trains
  and its original service start time. Draft creation rechecked OSRD conflicts and the
  source snapshot. This was a local Admin test; no planner review was submitted.
- A further live complete-calendar test used local timetable 153449 with three synthetic
  source services. Two requested service patterns produced 20 joint candidates; 16 passed
  pathfinding, simulation, and zero-new-conflict checks. The selected option created draft
  scenario 22/timetable 153497 with nine trains while the source scenario retained its
  original three. A change-only test then evaluated seven bounded retiming candidates for
  `Local regional test service`; six passed. Selecting the +60-second option created draft
  scenario 24/timetable 153513, where OSRD displays the service at 23:56; source scenario
  20 remains at 23:55. Draft materialization now checks that each copied service retains
  its requested start time. These remain local synthetic Admin tests and were not submitted
  as planner reviews.
- The conflict-recovery loop was exercised against the same synthetic source timetable.
  Retiming `Local regional test service` by +300 seconds produced one new OSRD spacing
  conflict with `Conflict target discovery demo-01`; the response named both affected
  schedule IDs. This exposed a UI gap: change-only plans did not show the existing bounded
  recovery action. The planner now offers that action after retiming results. In a
  browser-driven rerun, it added the conflicted service to the eligible set; OSRD evaluated
  20 options and 18 passed. Selecting a joint option created draft scenario 25/timetable
  153604, with departures changed to 23:54 and 00:01. A wider 49-candidate API search
  produced 36 zero-new-conflict options. After draft creation, the source revision and all
  three source departure timestamps were still unchanged. This verifies conflict
  identification, bounded multi-service recovery, and draft materialization through OSRD;
  a planner still authorizes which affected services may move and chooses a validated
  option.
- An authenticated Brave tab is open to the local OSRD deployment as `OSRD Demo Planner`.
  The local mock provider has no password and grants the `Admin` role only for this
  machine-bound fixture stack. No authenticated participating-operator deployment,
  planner contacts, operator endpoints, study-assigned physical-network labels, or
  authorization/review-group configuration is available in the current context.
- Static Belgium, France, Netherlands, and Switzerland datasets remain unsuitable as
  substitutes for operator participation, active timetables, or reviewed infrastructure
  independence.

To start collection, connect an authenticated OSRD deployment and identify at least eight
participating, physically independent infrastructure groups with stable study labels.
Each group must have real calendar and hourly timetables and planners who will review
options. Deploy the review-pilot overlay with each assigned label, then collect at least
30 eligible multi-option choices per mode in every group. Do not initialize labels from
static datasets, count the synthetic smoke request, or fit a preference model before the
coverage and registry gates pass.
