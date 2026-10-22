# Training data and model readiness

**Current model status (2026-10-17):** v34 is the current multi-network research artifact
for coupled service insertion and retiming. It was trained on 15,120 Swiss/Belgian OSRD
episodes, selected using 6,720 validation episodes from separate sites, and evaluated on
10,080 episodes from newly collected final holdouts (four new sites per network). Test
conflict AP is 0.8994 versus 0.8127 for logistic regression; Brier is 0.0565 versus
0.0752; arrival-change MAE is 7.60 seconds versus 401.79 seconds for zero-change.
However, the worst of 48 groups is -0.214 AP versus logistic, well below the required
+0.05 margin. Learned ranking remains disabled.

**Data-integrity correction (2026-10-17):** the OSM replay collector was passing RailJSON
track offsets (metres) directly to Editoast, whose track-offset API uses millimetres. A
corrected Belgium and Switzerland smoke replay now records explicit `track_offset_unit=mm`
and station operational-point offsets in millimetres; both pass OSRD pathfinding and
simulation. In the corrected two-seed samples, a 60-second station dwell changed new
conflicts 4→3 / 4→4 in Belgium and 2→3 / 3→4 in Switzerland. These tiny samples are
integration checks only. Historical OSM switch cohorts created by the old collector used
the wrong intended coordinate scale and must not support route-extent, station-stop, or
cross-network promotion claims. Their labels describe the paths OSRD actually simulated,
but not the intended OSM-derived route endpoints. The v34/v35 OSM holdout metrics above
are retained as historical results and are under re-evaluation; corrected geographically
split OSM data and a fresh model fit are still required.

Action-cost weight 0.1 was selected using validation only. On the new holdout the GNN
matches the exact simulated objective in 41.0% of 960 Swiss contexts and 84.0% of 480
Belgian contexts. Its ordering finds the first zero-new-conflict plan in 1.01 / 2.00
simulations on average; 80 Belgian contexts have no accepted candidate in the seven-action
grid. Minimum-edit ordering reaches a lower-cost accepted plan in Switzerland, while
the GNN uses fewer retrospective simulations. The model-selected action still averages
0.025 / 0.338 new conflicts per Swiss / Belgian context before OSRD validation. These
call counts are retrospective, since every alternative was simulated to create labels.

v34 estimates impacts and orders bounded candidates; it does not invent routes, station
stops, or complete timetables. See the [v34 model card](../../ops-intelligence/models/trained/service-plan-v34-multi-network-pristine-holdout/MODEL_CARD.md), [validation-only cost calibration](../../ops-intelligence/models/trained/service-plan-v34-multi-network-pristine-holdout/coupled-plan-cost-weight-validation.json), [new Swiss holdout evaluation](../../ops-intelligence/models/trained/service-plan-v34-multi-network-pristine-holdout/swiss-pristine-heldout-coupled-plan.json), and [new Belgian holdout evaluation](../../ops-intelligence/models/trained/service-plan-v34-multi-network-pristine-holdout/belgium-pristine-heldout-coupled-plan.json).

**Plan-level GNN research artifacts (17–20 October 2026):** the early
[`plan-gnn-v1`](../../ops-intelligence/models/trained/plan-gnn-v1/MODEL_CARD.md)
ranker used a 31,920-episode corpus with no stop, dwell, or route-choice variation.
Its previously inspected test cohorts were diagnostic rather than untouched.
The newer
[`plan-gnn-v3-route-aware`](../../ops-intelligence/models/trained/plan-gnn-v3-route-aware/MODEL_CARD.md)
adds a narrow station-stop and dwell grid: 384 training, 96 validation, and 288
diagnostic test episodes across 4, 1, and 3 scenario groups respectively. It
covers two frequencies, two departure phases, 512 stop episodes at eight passenger
operational points, and two nonzero dwell patterns—but zero route-waypoint actions
and no retiming. Pairwise concordance is 0.941 on validation and 0.977 on the small
diagnostic test; the held-out corridors are within the same two imported OSM
networks, not independent operator networks. Aggregate delay-head error remains
high. These synthetic counterfactual results are exploratory and do not demonstrate
planner benefit or operator preference.

Both plan-level scorers remain research-only and disconnected from planning APIs.
They order supplied candidate replay graphs; they do not create service routes,
stop patterns, or timetables and cannot establish feasibility. Every candidate still
requires complete OSRD pathfinding, simulation, and conflict validation. The v3
artifact improves stop/dwell representation but does not solve the requested
service-requirements-to-timetable generation problem. See its model card for data
hashes, metrics, and safeguards.

**Data needed to complete the model:** the currently available public actuals
condition synthetic replays but cannot tell us what a different timetable would
have caused. Remaining essential sources are operator-authorized historical
timetable snapshots and realized movement outcomes, plus prospective decisions on
untouched requests. The planner now supports explicit submission of immutable
candidate-set review records, including an optional categorical reason; the local
SQLite journal can feed these records to the review-event exporter. This is an
initial preference-capture path, not a record of acceptance or operational outcome.
It will require enough independent planner choices and infrastructure/timetable
contexts before preference fitting. Route/stop patterns, rolling-stock constraints,
cancellations/freight, and corridor-scale interactions also need paired OSRD
generation and validation.
Until these sources are available, this is a trained research ranker, not a
complete operational planner model.

The OSM replay collector now accepts matched route-waypoint action sets, including
an unconstrained `none` option, while preserving the same seed and scenario group
for paired comparisons. This provides a collection path for route-conditioned
training data, but it has not yet been run to expand the corpus; the existing
route-waypoint smoke cohort still contains only two episodes.

The repository also contains 200 route-waypoint episodes from the synthetic medium
infrastructure. Its 40-pair TC0/TC1 route-choice evaluation found identical new-conflict
counts and positive arrival delay for both routes; the model-selected route matched the
oracle only 50% of the time, and operational ranking eligibility remained false. These
episodes show waypoint coverage, not route-sensitive labels. The next route-data effort
must find and validate matched contexts where the feasible route alternatives produce
meaningfully different whole-timetable outcomes, then repeat on independent OSRD
infrastructures. See the
[`v15 route-choice model card`](../../ops-intelligence/models/trained/service-plan-v15-route-choice/MODEL_CARD.md)
and [evaluation](../../ops-intelligence/models/trained/service-plan-v15-route-choice/route-choice-evaluation.json).

**Historical model progression:** RIDE operational-delay data, Swiss Actual data, and derived
profiles are stored locally and excluded from Git. Twenty-four CC0 Infrabel monthly
event files (January 2024–December 2025) have been acquired, normalized, and checksummed
locally. The 2025 profile adds 1,178,119 observed
train-origin delay samples. The action-impact corpus now includes randomized and
controlled first-departure phases, route variants, stop and frequency choices,
full-year-profile-conditioned disturbances, plus geographic switch routes from Belgium
and Switzerland OSM imports. V25 used 4,725 Belgian training / 20 validation / 3,300
held-out episodes across 122 / 1 / 71 groups; no Swiss episode was used in training.
Aggregate test conflict AP is 0.962 versus 0.895 for logistic; Brier score is 0.089
versus 0.130. On the independent Swiss 160-pair 120s-versus-600s frequency holdout,
conditioned by one day of status-filtered Swiss Actual data, it chooses the lower-conflict
frequency 92.5% of the time and averages 0.684 fewer new conflicts than random across
four locations; positive arrival delay is unchanged. The minimum held-out-group AP gain
is -0.0176, so model-guided ranking remains disabled. V26 adds 880 southern Swiss
training episodes and 80 episodes from a separate southern Swiss validation site, while
preserving all 71 previous held-out groups. V26 completed with 5,605 training, 100
validation, and 3,300 test episodes across 144 / 3 / 71 groups. Pooled test conflict AP is
0.978 versus 0.903 for logistic and Brier is 0.0662 versus 0.1278. Arrival MAE is 3.27 s
versus 38.48 s for zero-change. The per-group promotion gate still fails: one Midi–Nord
adjacent-track group is -0.496 AP versus logistic. On the frozen 160-pair Swiss frequency
choice benchmark, v26 repeats v25's 92.5% accuracy and 2.950 versus 3.634 mean conflicts.
V27 adds a second Swiss seed using the 62-day profile and completes with 6,485 training,
180 validation, and 3,300 test episodes across 144 / 3 / 71 groups. V27 pooled test
conflict AP is 0.978 versus 0.907 for logistic and Brier is 0.0686 versus 0.1266. Arrival
MAE is 2.66 s versus 38.48 s for zero-change. The worst test-group AP difference is
-0.0048, so the required +0.05 margin still fails. On the frozen 160-pair Swiss frequency
choice benchmark, v27 selects the lower-conflict frequency in 92.5% of pairs, averaging
2.950 conflicts versus 3.634 for random. Ranking remains disabled. Labels remain OSRD
simulation outcomes, not real Swiss operational impact labels.
A Swiss Actual data v2 archive is available from 2016. V28 acquired 15 monthly archives
from 2025-07-13 through 2026-09-30 and built a rail-only profile from 445 daily CSV
members. It scans 1,054,186,238 rows, filters `PRODUKT_ID=Zug`, and retains bounded
250,000 delay and 250,000 dwell samples from source populations of 6,117,227 `REAL`
first departures and 57,518,355 dwell observations. A separate rail-only profile from
2026-10-14 conditions held-out OSRD episodes. These are empirical context distributions,
not Swiss counterfactual impact labels.

**Modal-data correction (2026-10-17):** Swiss Actual v2 contains multiple transport
products. The v25-v27 profile adapter did not filter `PRODUKT_ID`, so those Swiss
profile-conditioned cohorts mixed rail and non-rail disturbances and are not valid as
rail-only conditioning benchmarks. The rebuilt 2026-10-14 profile filters `Zug` and
contains 14,026 `REAL` first-departure samples after excluding 2,423,561 non-rail rows.
V28 uses the corrected one-day profile for held-out replay context and a separate
15-month rail-only profile for training context; previous mixed-mode profiles and models
remain as historical artifacts and are excluded from the v28 fit.
The Belgium-trained corpus spans six OSRD fixtures and one Belgium OSM import; v24 and
v25 have a separate held-out Swiss OSM topology test, with no Swiss episodes in training.
V26 is the first multi-network fit, using southern Swiss simulations for training and
validation while retaining northern Swiss test groups. V27 adds a second seed conditioned
on the two-month Swiss profile and is the current fit. This is simulated cross-network transfer, not
observed-outcome validation. Most Swiss historical actual-data files have not been
acquired. The publisher
states that records may be measured actuals, estimated actuals, or the last forecast (and
some journeys are absent); the adapter preserves per-event status and only uses `REAL`
first-departure records in this profile. The files are useful for observational forecasting and
external plausibility checks, not counterfactual schedule-change labels. See the
[official actual-data schema and quality notes](https://opentransportdata.swiss/en/cookbook/historic-and-statistics-cookbook/actual-data/)
and [archive](https://archive.opentransportdata.swiss/actual_data_archive.htm).
**Decision:** keep observational delay forecasting separate from OSRD-generated impact
labels; neither data source substitutes for the other.

## Long-term planning-assistant goal

The end goal is to help planners create and revise timetables and routes, then quickly
show the network-wide consequences of a proposed change: which other trains, stations,
and resources are affected; where delay or conflicts may grow; and which feasible
alternatives best meet the planner's priorities. The system should propose and compare
drafts, with OSRD pathfinding, simulation, and conflict detection remaining the authority
for feasibility. A planner reviews and applies the selected draft.

The RIDE GNN is an operational delay-forecasting model; it does not take a timetable edit
as input. A separate OSRD replay-impact GNN now estimates service-level conflict risk and
arrival deltas for an already specified service insertion, stop pattern, and bounded
retiming action. It does not construct a timetable or learn which route to choose. The
service planner can enumerate up to five topological alternatives for track-offset paths
and have OSRD validate them. Its current training and evaluation labels are synthetic OSRD
simulation outcomes, and its conflict-ranking gate fails. More training on observational delay data alone cannot
provide counterfactual labels. Continue generating paired OSRD baseline/treatment
scenarios, improve the model against frozen references, and add independently sourced
infrastructure and operational outcomes. Always send promising candidates through full
OSRD validation before presenting them as feasible.

The second full-year Belgian profile was used to collect four new 40-episode service-plan
cohorts (small, Y-branch, ETCS, and nested-switch fixtures). The resulting v13 fit keeps
the previous validation group and all 20 held-out scenario groups unchanged: 1,725 train,
20 validation, and 780 test episodes. Test conflict AP is 0.9862 versus 0.9530 for
logistic; Brier is 0.0693 versus 0.1155. Arrival-delta MAE is 3.67 s versus 46.45 s for
zero-change, but these remain synthetic simulator labels. The minimum per-group AP gain
fell to 0.0018 (from 0.0031), so the +0.05 promotion gate still fails. Artifact:
`ops-intelligence/models/trained/service-plan-v13-infrabel-2025-expanded`. Do not use the
training profile as evidence that the model has learned real Belgian route effects: no
Belgian track topology or event IDs are mapped to these OSRD fixtures.

The v15 refit adds 80 training-only route-conditioned episodes from a second matched seed
on two synthetic platform routes. A separate 40-pair TC0/TC1 route-choice cohort is held
out by scenario group, with exact baseline and perturbation matching verified. V15 has
1,845 train / 20 validation / 860 test episodes across 50 / 1 / 22 groups. Test conflict
AP is 0.9844 versus 0.9527 logistic and Brier is 0.0670 versus 0.1153; the minimum group
AP gain is +0.00094, so the +0.05 gate fails. On the held-out route pairs, both paths have
identical per-train conflict and arrival outcomes; model choice is 50% and has no measured
benefit. These pairs validate the route-action data format, not network transfer. Artifact:
`ops-intelligence/models/trained/service-plan-v15-route-choice`.

V16 keeps v15's training and validation groups fixed and adds a 40-episode OSM
adjacent-track group only to the test set (1,845 train / 20 validation / 900 test; 50 / 1
/ 23 groups). This segment connects `1122923669-0` to `116000117-0`; full OSRD pathfinding
and simulation completed for all episodes. Test-group conflict AP is 1.000 for both the
GNN and logistic reference; Brier is 0.197 versus 0.561. Arrival MAE is 72.86 s, worse
than zero-change at 20.44 s. The zero per-group AP improvement keeps ranking disabled.
The route-choice holdout remains 50% with tied outcomes. This is a short topology holdout
within the same OSM import, not independent-network transfer. Artifact:
`ops-intelligence/models/trained/service-plan-v16-osm-adjacent-holdout`.

## Why there are two training tracks

The existing `osrd-ops-plan-feasibility-model/v1` logistic ranker learns whether an
OSRD-validated candidate passes pathfinding, complete simulation, and zero-conflict
checks. Its labels come from the append-only planning audit journal. The local journal
currently contains no completed plans, so it cannot fit or evaluate that ranker yet.

Public railway datasets contain schedules, realized or forecast service times, delays,
and punctuality. They do not contain OSRD counterfactual outcomes for the candidate
schedule edits this ranker sees. Do not relabel an observed delay or a punctuality
threshold as candidate feasibility.

The repository also has a separate, non-LLM RIDE event delay forecaster. It can supply
delay context to future candidate scoring, but it cannot accept plans, validate timetable
feasibility, or replace OSRD simulation. The coupled-plan impact model described above
uses matched OSRD counterfactual simulations for candidate labels; observed delay records
alone cannot provide those labels.

## Dataset assessment

| Source | Training value | Limits and handling | Decision |
|---|---|---|---|
| [RIDE Gold Lite](https://huggingface.co/datasets/orailix/ride-gold-lite) | Belgium passenger operations; fixed January 2023–December 2024 training snapshots (15,000) and a January–December 2025 held-out test period (3,000 snapshots). Provides tabular, sequence and graph arrays, masks, metadata and normalization/scheme files. The core test table has 623,314 rows and 15 future-event delay targets. | CC BY 4.0. Preserve attribution to RIDE, Infrabel and Open-Meteo. The Lite release is listed at 26.5 GB across all model representations; fetch only the tabular or GNN representation needed for an experiment, pin the dataset revision and per-file SHA-256, and do not download the whole tree. Use provider masks and schema. Do not use the 2025 split for model or hyperparameter selection. | **First public forecast source and GNN benchmark.** Compare a graph model with the current tabular baseline on the same Gold Lite core/task. |
| [Infrabel monthly raw punctuality files](https://opendata.infrabel.be/explore/dataset/stiptheid-gegevens-maandelijksebestanden/) | Belgian passenger arrival/departure events by day and train number, with planned/actual times, signed delay seconds, stop IDs and line IDs. Monthly files run from 2014; the publisher lists CC0 and attribution to Infrabel. | January–December 2024 contains 23,114,728 events; January–December 2025 adds 22,427,357 events (monthly normalized counts are recorded in `data/rail/infrabel-punctuality/events/manifest.json`). The 2024 profile has 1,218,110 origin delays and 20,814,142 dwell samples; the 2025 profile has 1,178,119 origin delays. These marginals condition OSRD replay inputs, not OSRD outcome labels. IDs are not mapped to OSRD topology or RIDE journey IDs. | **Two annual profiles acquired and normalized.** Used to condition paired OSRD insertion, stop and frequency episodes; topology mapping remains. |
| [OpenStreetMap via Geofabrik Belgium](https://download.geofabrik.de/europe/belgium.html) and OSRD's [osm-to-railjson converter](https://osrd.fr/en/docs/guides/osm_infra/generation/) | Dated 2026-10-15 Belgium OSM extract (665 MiB), converted to 15,571 tracks, 13,479 switches, 540 operational points. New geographic cohorts add 480 episodes across 12 southern switch locations and 160 episodes from four northern sites. Phase training adds 480 at 180s and 960 at 60s/120s; a separate fresh northern holdout contains 160 matched 60s-versus-120s pairs. Frequency training adds 960 at 120s/600s, with a separate 160-pair northern holdout at those frequencies. | ODbL 1.0; retain OSM attribution and source snapshot/hash. Switch links are inferred from geometry; signals are generated. This is one OSM-derived Belgian infrastructure and all outcome labels come from OSRD simulation, conditioned by Infrabel delay profiles. Same-track, short adjacent-track and switch-level route simulations passed. Long Midi–Nord pathfinding still fails; geographic crops did not produce full corridor labels. | **Useful within-network holdout; v23 gets 90% frequency-choice accuracy and 0.566 fewer conflicts than random on 160 fresh pairs. The minimum held-out AP gain is negative, so ranking remains disabled.** |
| [OpenStreetMap via Geofabrik Switzerland](https://download.geofabrik.de/europe/switzerland.html) and OSRD's [osm-to-railjson converter](https://osrd.fr/en/docs/guides/osm_infra/generation/) | PBF snapshot dated 2026-10-15 (548,527,247 bytes; SHA-256 in `data/rail/osm-switzerland/source-manifest.json`), converted to 19,313 tracks, 17,694 switches, 790 operational points, and imported as OSRD infra 17. V28 uses eight southern sites for 1,680 training and 240 validation episodes and four northern sites for 960 held-out episodes across six service frequencies (120/180/240/300/420/600 seconds). | ODbL 1.0 and `© OpenStreetMap contributors`. Generated signals and inferred switch topology require operational review. All impact labels are paired OSRD simulations, not real Swiss conflict labels. Northern locations are held out within the same Swiss OSM import; this is not an independent operator-network test. | **V28 demonstrates useful offline conflict ranking against a fixed 300-second choice, but ties a fixed 600-second policy. Service ranking remains gated off; the model does not account for requested service intensity.** |
| Coupled service-plan replay corpus (Swiss + Belgian OSM) | The v34 corpus contains 31,920 paired OSRD episodes across 156 spatial/frequency/phase groups. Train: 10,080 Swiss + 5,040 Belgian; validation: 3,360 per network; final test: 6,720 Swiss + 3,360 Belgian from four newly collected sites per network. Each request has seven matched alternatives covering no retiming, one-service retiming, and five coordinated all-service shifts, with 300s/600s service frequency and three phases. Cohort manifests bind OSRD replay outputs to SHA-256 checksums. | Belgian data use Infrabel 2025 origin-delay context. Swiss training/validation use 2025-07 through 2026-09 rail-only Swiss Actual context; the new Swiss test uses a separate 2026-10-14 rail-only profile. Delay and dwell distributions condition OSRD simulations but do not provide counterfactual conflict labels. The new final test is isolated from the v34 fit and validation, but the group-level promotion gate still fails. | **Complete available training/validation/test corpus for the current coupled-plan research model.** Model: [v34 card](../../ops-intelligence/models/trained/service-plan-v34-multi-network-pristine-holdout/MODEL_CARD.md); datasets: [`datasets/replay-service-plan-v7`](../../ops-intelligence/datasets/replay-service-plan-v7). |
| [Swiss Actual data v2](https://data.opentransportdata.swiss/en/dataset/ist-daten-v2) and [SBB previous-day target/actual comparison](https://data.sbb.ch/explore/dataset/ist-daten-sbb/) | The archive provides monthly ZIP files from 2016 through the latest complete month. V28 uses 15 archives spanning 2025-07-13 through 2026-09-30 (445 daily CSV members; 1,054,186,238 rows scanned). `PRODUKT_ID=Zug` filtering excludes 974,785,755 bus and other non-rail rows. The rail-only bounded profile contains 250,000 delay and 250,000 planned-dwell samples drawn from populations of 6,117,227 `REAL` first departures and 57,518,355 dwell observations. A separate 2026-10-14 rail-only daily profile conditions held-out replays. | Provider states values can be measured actual or the latest forecast, and trips without real-time data can be absent. The v28 adapter retains only `REAL` first-departure values for delay samples and scheduled arrival/departure differences for dwell context; forecast, estimated, unknown and empty statuses are excluded. The 2026-10-12 malformed file remains excluded. Checksums and archive/member provenance are in `data/rail/swiss-actual-v2/source-manifest.json`. V25-v27 did not filter transport product and are historical mixed-mode results; their profile-conditioned cohorts are excluded from v28. Swiss actuals condition synthetic OSRD simulations and are not mapped counterfactual labels. | **Required 15-month rail-only context is acquired and used in v28 training.** The selected archive period is not the full history since 2016; additional seasons/operators and real-network evaluation remain future work. |
| [Swiss GTFS Static](https://opentransportdata.swiss/en/cookbook/timetable-cookbook/gtfs/) | Nationwide timetable-year schedules, trips, stop times, calendars, stops and routes. Supports schedule context and Swiss timetable pattern exploration. | Static timetable is not realized operation or delay labels. IDs can change when feeds regenerate, and GTFS lacks some provider timetable detail. Retain feed snapshot and service date. | **Schedule context only**, not a supervised delay target by itself. |
| [Infrabel monthly freight punctuality](https://opendata.infrabel.be/explore/dataset/stiptheid-van-de-goederentreinen/) | Freight destination punctuality summaries with CC0 terms. | Monthly aggregate target is too coarse for per-event propagation or schedule-edit feasibility. | Aggregate monitoring only. |

Dataset facts and constraints were last checked against the linked source pages on
2026-10-17. On 2026-10-13, the Swiss Actual data v2 publisher reported a production
disruption affecting the 2026-10-12 file, with malformed lines due to missing line breaks.
Exclude that date from ingestion until a corrected source file is available and validated.
The archive confirms monthly history from 2016 through the latest complete month; terms
and schema must be rechecked before acquisition or redistribution.

## Graph model inputs now available

RIDE already publishes the structural inputs we need for a first heterogeneous graph
model; the graph is not something we need to synthesize from the current MLP's feature
matrix. The Gold Lite `gnn/` representation consists of PyTorch Geometric heterogeneous
graph shards for train and test, plus `feature_spec.yaml` and `normalization.yaml`. Its
feature groups describe train nodes, station nodes, station-to-station edges, past-event
edges and known-future-event edges. The listed fields include train operator/service
type and snapshot-time encodings; station coordinates/weather/active-train delay
aggregates; directional station-link distance/angle/traffic summaries; past event
planned-time offsets and observed delay; and future scheduled event offsets/type. This
is a heterogeneous **event/station interaction graph**, not an OSRD RailJSON track,
signal, block-occupancy or route-feasibility graph. It supports delay forecasting; it
does not certify a proposed schedule edit.

The public Gold Lite file manifest lists these digests for the published GNN metadata
and graph shards. The files are present at dataset revision
`e5c1ed1126bca1784eb6ea93816dd6cc89afa544`; verify each digest after download and keep
the revision and manifest with every run:

| File | SHA-256 |
|---|---|
| `gnn/feature_spec.yaml` | `28f636af3bf5ebac9df7190648b57d42bea795558a4842985e5f95fda9f8f9e3` |
| `gnn/normalization.yaml` | `8ebc3a30585ab2f4dd0f83eb0505375cf8d8aaf7d52694bd2f7bbaecc756fcfb` |
| `gnn/train/graphs_part_00000.pt` | `df9fef3fc0c72a47e6fea534afd51e2c959864a53de737f43f1d08e25a81a0ef` |
| `gnn/train/graphs_part_00001.pt` | `930cb01e2c0bc9b6da568b9f75b72791487992520976413aa4392459dbd9f1bc` |
| `gnn/test/graphs_part_00000.pt` | `084b9f070c0d68814be9be8ac37c62646b6eb834f8379b23d3d4e0cb21396b9c` |

### Obtain the GNN files

RIDE's supported all-GNN-component download is run from a checkout of the upstream
RIDE repository root:

```sh
python -m scripts.dataset.download.download_dataset --target gold_lite_gnn
```

It writes `data/gold/lite/gnn/` and includes both training and held-out graph files.
For controlled model development, fetch only the training files from Hugging Face at
the pinned revision in the manifest. The intended allowlist is
`gnn/feature_spec.yaml`, `gnn/normalization.yaml`, and
`gnn/train/graphs_part_00000.pt`, `gnn/train/graphs_part_00001.pt`; do not request
`gnn/test/**` until the model and tuning decisions are frozen. Public access does not
require a Hugging Face key; `HF_TOKEN` can be supplied for authenticated transfer if
needed. Use the Hub's resumable Xet-backed transfer, check available disk space first,
and verify every SHA-256 before loading. The shard sizes need to be measured from the
provider metadata rather than inferred from the 26.5 GB whole-release size.

The `.pt` files use Python/PyTorch serialization. Treat them as executable/untrusted
pickle input: accept only the pinned files after digest verification, load in an
isolated training environment, and do not load arbitrary user-provided checkpoints.
The published normalization values and feature definitions are part of the dataset
contract; do not refit normalization on validation or test data.

RIDE publishes its model-specific builder, trainer, evaluation script and fixed Lite
configuration. Its reference scale is hidden width 128, six layers, mean heterogeneous
aggregation, batch size 32, dropout 0.15 and 34 epochs. The local edge-aware GATv2
trainer is an initial implementation at that scale, not yet an exact reproduction of
the published model. Keep the distinction explicit until its layer types, feature
mapping and target inversion are matched against upstream source.

The first local implementation should add an optional `gnn` dependency extra (PyTorch
and PyTorch Geometric), a lazy shard reader, schema/digest validation, and an isolated
`ride_gnn` training entry point. Keep the serving/API environment free of these heavy
training dependencies. A graph example is not an MLP row: represent each snapshot's
active train and its station/event neighborhood as a typed heterogeneous graph, encode
node and relation features, pool to the prediction train/event, and emit the same
15-horizon delay target (plus masks) used by the Gold core. Preserve a stable join key
for each prediction, at minimum snapshot time, train ID, service date, and event/horizon
identity, and assert that graph outputs map one-to-one to target rows.

Compare GNN, current MLP and linear baseline on identical snapshot groups and identical
valid target masks. Split by service date/snapshot group, never by individual graph or
event; fit preprocessing only on train. If the published train graphs lack enough keys
for leakage-safe internal validation, build the representation from RIDE Silver and the
Gold core using the official builder, creating chronological fit/early-stop/calibration
groups within the training period. Keep the 2025 test graphs and test evaluation table
sealed until architecture, thresholds and hyperparameters are frozen. Score per-horizon
MAE, RMSE, signed bias and valid counts, with service-date clustered uncertainty and
the same final test protocol as the tabular models.

The 2025 test was kept sealed during model selection and has now been evaluated once
for the frozen checkpoint; do not tune against that result.

### Initial full GNN result on the 2025 test

The initial edge-aware GATv2 was trained on all 15,000 provider training snapshots for
34 epochs, with checkpoint selection on a chronological validation split. Its best
checkpoint (epoch 25) was evaluated once on the pinned 3,000-snapshot 2025 test shard:

| Model / reference | MAE (s) | RMSE (s) | Valid targets |
|---|---:|---:|---:|
| RIDE GNN, future delay-change prediction | 83.28 | 205.18 | 6,528,133 |
| No-change persistence reference (predict zero delay change) | 97.48 | 236.87 | 6,528,133 |

This is a 14.6% lower MAE and 13.4% lower RMSE than the no-change reference on this
test split; the GNN improved MAE at all 15 forecast ranks. The benchmark is still
limited: this local GATv2 is not an exact upstream RIDE reproduction, there is no
network-transfer test, and it predicts observed delay change rather than effects caused
by proposed timetable edits. It does not establish planning utility. The graph test
shard and JSON evaluation report remain in ignored local data/artifact directories; the
report includes per-rank metrics and pinned source/checkpoint hashes.

For a lower-cost parser/loader smoke check, a third-party [RIDE representative
sample](https://huggingface.co/datasets/ano6060/anonymous-ride-sample) says it contains
the same pipeline's Gold GNN representation, built from 200 January 2023 training
snapshots and 100 test snapshots; the full sample is listed at 724 MB. It explicitly
is not for benchmark results. Its public card does not expose a clear license in the
metadata viewed here, so confirm redistribution/use terms before downloading; prefer
the official RIDE Gold Lite release for any reported model result.

Next, add an OSRD-native representation separately: derive directed traversable
segments from the pinned RailJSON infrastructure (track-section endpoints, switches,
routes and resource ranges), attach time-indexed train-event/resource-occupancy
features, and validate each graph build against OSRD path/simulation outputs. Keep
provider station IDs, RIDE event graph IDs and OSRD RailJSON IDs in separate typed
namespaces with explicit mapping confidence; never infer that they are interchangeable.
The RIDE GNN can establish whether a GNN improves this public delay benchmark, while
OSRD-generated replay/counterfactual data and the OSRD validator remain necessary for
planning and operational claims.

## First forecast experiment

### Task and target

At a prediction snapshot, forecast signed arrival/departure delay in seconds at each of
the next 15 train events. The first benchmark should use the RIDE Gold Lite tabular
representation and its provided target-validity mask. This is an observed-operation
forecast, not an estimate of the causal effect of changing a timetable.

### Input and leakage rules

- Pin the RIDE repository revision, file paths, hashes, license, attribution, source
  commit, and local transform version in a generated data manifest.
- Download only `tabular/train/{x,y,y_mask,md}.npy`, `tabular/test/{x,y,y_mask,md}.npy`,
  `tabular/scheme.yaml`, and `tabular/normalization.yaml`; do not fetch sequential/GNN
  tensors for the first experiment.
- Read feature and target meaning from `scheme.yaml`; never infer column order from
  array shape. Load NumPy files with `allow_pickle=False` unless source inspection
  proves the metadata encoding requires a safer explicit alternative.
- Apply the published training transform and preserve its normalization metadata.
  Fit all learned parameters only on the internal training partition. Keep the fixed
  2025 benchmark arrays unused until a single final evaluation.
- Derive grouped chronological partitions from `md.npy`, keeping all snapshots from a
  service date together. Rows before the configured cutoff fit the model; later
  service dates are divided chronologically into early-stop, interval-calibration, and
  internal-evaluation groups. Report each date range and row count. Do not randomly
  split snapshots of a service date across partitions.
- Exclude invalid targets using `y_mask`; report valid target counts separately at
  every forecast horizon. Do not impute a missing target as zero delay.

### Model sequence

1. Reproduce the persistence reference on the official 2025 test table (already
   present locally) and retain its per-horizon MAE, RMSE, bias, and valid counts.
2. Fit a transparent regularized multi-output linear regression on the tabular arrays
   as the first trainable baseline. Optimize only observed targets under the mask;
   compare against persistence and a training-mean reference.
3. Evaluate the implemented CPU-friendly feed-forward MLP on the same bounded tabular
   sample as the linear model. Tune architecture and regularization on chronological
   validation only. Keep one model artifact per immutable dataset/feature schema.
4. Report MAE, RMSE, signed bias and target coverage for each horizon (1–15), plus
   macro-by-horizon aggregates, service-date slices, and bootstrap confidence
   intervals clustered by service date. Report calibration/interval coverage before
   exposing uncertainty to the planner.
5. Run the held-out 2025 test once after model and hyperparameters are frozen. Compare
   to the persistence baseline and publish the manifest, report, source hashes, code
   commit, seed, and model hash.

No minimum improvement is claimed in advance. A model that does not beat persistence
reliably remains an experiment and is not integrated into candidate ranking.

## Planner ranker data plan

Continue to use the existing planner audit exporter and candidate ranker only for the
feasibility target. Before training, collect candidates from multiple OSRD scenario
groups with both accepted and rejected outcomes in the training groups. Exercise both
`objective_first` and `diverse` validation orders, record the exact candidate proposal
pool and selection policy, and split by scenario/infrastructure rather than candidate
row. Report top-k validation yield against objective order and the oracle, with group
counts and uncertainty. Data gathered only from candidates the current policy chose
for validation remains selection-biased; the score can prioritize OSRD validation only.

Operator utility or preference training needs a separately defined outcome source
(e.g. reviewed/accepted plans with policy context); neither RIDE nor timetable
punctuality data supplies that label.

The offline planner-choice preference trainer now exports v5 review events with
separate patronage boarding/alighting coverage and mapping-quality features, then fits a
train-normalized regularized quadratic listwise ranker. This permits conditional learned
trade-offs between demand coverage and operational costs while preserving a readable
coefficient artifact. It remains strictly research-only and does not affect the live
planner. For a patronage-specific fit, `--require-patronage` requires at least 30 complete
profile-backed choice sets in each frozen infrastructure-held-out split; evaluate against
the deterministic order and report infrastructure-level metrics before considering any
live use. Historical v3/v4 events remain consumable, with patronage fields treated as
missing.

## Implementation checklist

- [x] Add a pinned, checksum-verifying RIDE tabular training-split fetch command; default
      output stays under ignored `ops-intelligence/data/rail/`. It deliberately does not
      download held-out arrays or sequence/graph tensors.
- [x] Add a loader that validates array ranks/shapes and pinned file digests before training.
- [x] Add chronological service-date validation and deterministic bounded sampling.
- [x] Add a masked multi-horizon linear reference and save model/provenance/report artifacts.
- [x] Add an optional tabular MLP training path with train-only feature scaling and masked loss.
- [x] Add best-checkpoint selection and early stopping on the chronological validation partition.
- [x] Add p10/p50/p90 quantile heads to the MLP and interval/pinball diagnostics.
- [x] Add split-conformal interval expansion with per-horizon offsets recorded in the artifact.
- [x] Integrate disjoint service-date early-stop, calibration, and evaluation groups into RIDE training.
- [x] Add a synthetic smoke fixture with train, early-stop, calibration, and test partitions.
- [x] Evaluate calibrated coverage and quantile crossing rates on real held-out dates.
- [x] Compare linear and MLP runs on identical samples before selecting either model.
- [x] Retrieve the RIDE GNN feature schema and normalization metadata; inspect actual serialized PyG objects and labels before designing our own representation.
- [x] Add an optional PyTorch/PyG extra and a pinned train-only GNN data fetcher; it verifies the published digests and excludes the 2025 test shard.
- [x] Fetch both pinned training graph shards (9.2 GB total) and verify their hashes.
- [x] Inspect graph node/relation types, edge feature shapes, labels, and the snapshot/train/service-date/future-rank join on representative graphs; each checked future edge had one finite target and a unique event key.
- [x] Implement an initial edge-aware heterogeneous GATv2 with date-grouped early stopping and a seconds-scale validation report; this is not yet the exact upstream RIDE reproduction.
- [x] Train the initial GATv2 on all 15,000 training snapshots and save the best validation checkpoint (epoch 25 of 34).
- [x] Fetch the pinned 2025 test graph after model selection and evaluate once against the no-change reference; per-rank metrics and hashes are in the ignored local report.
- [ ] Resolve the observed train-node width discrepancy (32 tensor columns versus 31 declared columns) and match upstream feature transforms, layers, target inverse transform, and benchmark protocol.
- [ ] Compare the GNN with linear/MLP baselines on the same task and compatible snapshot/target protocol; do not use the 2025 test result for model tuning.
- [ ] Design an OSRD RailJSON graph adapter separately; RIDE's station/event graph lacks track/switch/signal/block resource semantics.
- [x] Fetch the pinned multi-gigabyte tabular training files and run the first fit.
- [ ] Compare with the existing 2025 persistence report without tuning on test metrics.
- [ ] Keep model serving disconnected from the planning API until the offline result,
      applicability limits and safety review justify integration.
- [ ] Separately restart OSRD, populate an infrastructure and timetable, and collect
      scenario-diverse audit records for the candidate-feasibility ranker.
- [x] Collect paired insertion episodes with mixed background routes and per-service
      OSRD block, route, and track-section paths captured before either simulation arm.
- [x] Check in train, validation, and held-out corridor datasets with manifests and
      episode-file SHA-256 digests so the insertion model can be retrained locally.
- [x] Train a shared-resource edge GNN for service-insertion conflict ranking and
      evaluate it against linear and departure-gap references on a held-out corridor.
- [x] Add `small_infra` and a structurally different Y fixture as completely held-out
      OSRD tests and run cross-infrastructure insertion and retiming evaluations.
- [x] Add four route/density cohorts on the ETCS fixture (320 episodes) and collect
      fresh held-out cohorts on overlapping-routes and nested-switch fixtures.
- [x] Train/evaluate on 1,120 paired OSRD episodes across six fixtures (560/160/400
      train/validation/test); preserve all manifests and per-file SHA-256 digests.
- [x] Check the expanded five-group gate without tuning on these test outcomes; v5
      fails its 0.05 per-group AP margin (minimum 0.022), so inference suppresses
      threshold decisions for that artifact.
- [x] Fit positive-slope Platt calibration on validation groups and apply it at inference;
      held-out Brier improves from 0.108 to 0.054 overall, although the small-infra
      group remains worse than the linear probability baseline.
- [x] Disable arrival-delta estimates because the GNN loses to the zero-change baseline.
- [x] Train/evaluate a separate signed-retiming model against a held-out infrastructure;
      its 0.015 AP gain over linear is below the 0.05 margin, and its delay head fails.
- [x] Add an outcome-free, fail-closed API for advisory ranking of prebuilt single-service
      insertion alternatives; reject treatment outputs/IDs and require later full OSRD
      validation. The historical v3 fixture gate is narrow; v5 remains rejected.
- [x] Add a typed explicit-path service request that expands a UTC operating window and
      frequency into calendar departures, then validates the complete temporary timetable
      with OSRD simulation/conflicts, compares normalized conflict records against the
      source timetable, and preserves the source timetable.
- [x] Add bounded service-plan candidate generation for insertion-only, signed single-
      train, pairwise, and coordinated shifts of explicitly eligible trains; validate
      every complete candidate in OSRD and report only conflict-delta-safe feasibility.
- [x] Extend the collector and model features to v3 episode/v7 feature contracts for
      requested frequency insertion plus up to three signed edits to existing services.
- [x] Collect 360 checksummed paired episodes across six OSRD fixtures; train on four,
      validate on overlapping-routes, and hold out nested-switches for the first v7 fit.
- [x] Fit the v7 GNN and linear reference. Held-out conflict AP is 0.963 vs 0.943
      (below the 0.05 gate); arrival MAE is 22.84 s vs 37.15 s for zero-change. Keep
      learned ranking unavailable and treat arrival estimates as synthetic-only research.
- [x] Extend the OSRD collector with repeated service-insertion actions and versioned v6
      action features for cadence/window/count/stop summaries; retain inference support
      for the v5 feature schema.
- [x] Collect and checksum 108 frequency/stop-action episodes across medium, Y-branch,
      and ETCS routes; fit v6/v7/v8 models with grouped train/validation/test partitions.
- [x] Preserve the v8 artifact and holdout report; it has aggregate AP 0.732 vs 0.697
      linear, but its per-group minimum margin is 0.007 on nested switches, so it remains
      ineligible for ranking.
- [ ] Improve retiming and arrival-impact data before allowing that artifact to rank plans.
- [x] Collect paired labels for the new coupled service-plan candidates, including per-
      service arrival/departure deltas, new conflicts, infeasible path/simulation outcomes,
      and recorded signed action cost across six separated OSRD fixtures.
- [x] Acquire Infrabel January 2024 under its published CC0 terms, preserve source URL,
      ETag, retrieval time and SHA-256, normalize to Parquet, and quantify date-aware
      provider-label agreement and missingness.
- [x] Build a train-origin delay/planned-dwell profile and use it to condition 160 paired
      OSRD training episodes; preserve the profile checksum and original/applied
      perturbations in v4 episodes.
- [x] Acquire, normalize and checksum all twelve Infrabel 2024 monthly files; build a
      full-year profile with 1,218,110 origin-delay and 20,814,142 dwell samples.
- [x] Collect 160 paired episodes across four OSRD fixtures conditioned on the full-year
      profile, with manifest/profile checksums and v4 labels.
- [x] Refit v7 using Infrabel-conditioned episodes. Nested-switches AP is 0.954 vs 0.940
      linear (0.014 margin, below 0.05); arrival MAE is 9.58 s vs 37.15 s zero-change.
      Keep learned ranking unavailable; impact labels and topologies remain synthetic.
- [x] Refit with all 640 intended training episodes plus 20 validation and 20 test
      episodes. The annual-profile fit scores nested-switches AP 0.961 vs 0.929 logistic
      (+0.033, below the 0.05 promotion margin); arrival MAE is 39.73 s vs 37.15 s
      zero-change. Inference keeps ranking and arrival outputs disabled. This test group
      was evaluated by earlier fits and is diagnostic only, not an independent result.
- [x] Collect a new 40-episode annual-profile cohort on previously unused `tiny_infra`
      and evaluate the same frozen architecture once. Conflict AP is 0.922 vs 0.942 for
      logistic, so ranking remains gated off; arrival MAE is 16.35 s vs 102.97 s
      zero-change on this synthetic fixture only. Do not infer operational generalization
      from this tiny topology.
- [x] Collect 160 stop-pattern training episodes across four fixtures, balanced between
      60-second and 120-second requested dwells, plus a separate 40-episode tiny-topology
      holdout using a 90-second stop. The earlier v10 800/20/40 fit reached AP 0.991 vs 0.954 logistic
      (+0.037, below ranking promotion) and arrival MAE 13.38 s vs 113.70 s zero-change.
      Inference suppresses arrival estimates outside the validation infrastructure scope.
- [x] Expand action coverage with full-year-profile stop-pattern cohorts at 180-second
      frequency / 90-second dwell on small, Y-branch, and ETCS; add 120- and 300-second
      frequency / 90-second dwell cohorts on medium and Y-branch. Also add sparse
      600-second service-frequency cohorts across small, medium, Y-branch and ETCS, plus
      120/300-second stop-pattern cohorts on nested-switches. Preserve each cohort's OSRD
      SHA, source profile hash, scenario group, and episode digest.
- [x] Train v11 on the expanded 1,145-episode corpus with 20 validation and 120 test
      episodes across three scenario groups. Aggregate conflict AP is 0.963 vs 0.924
      logistic, Brier is 0.102 vs 0.141, and arrival MAE is 13.58 s vs 80.26 s
      zero-change. Per-group AP gains are +0.070 (medium sparse), +0.027 (nested-switches),
      and +0.010 (tiny topology); the required +0.05 margin fails. Ranking remains
      disabled; arrival inference is scoped to the validation infrastructure. Artifact:
      `ops-intelligence/models/trained/service-plan-v11-full-corpus`.
- [x] Repeat the 1,105-episode pre-nested fit with a second training seed on the same
      train/validation/test split. It keeps the ranking gate false; results show seed
      variability, especially on the tiny-topology holdout. The second-seed artifact is
      `ops-intelligence/models/trained/service-plan-v11-frequency-sparse-nested-seed42`.
- [x] Repeat the full 1,145-episode fit with seed 42, retaining the exact validation and
      three held-out scenario groups. The artifact is
      `ops-intelligence/models/trained/service-plan-v11-full-corpus-seed42` (epoch 9).
      Aggregate conflict AP is 0.966 vs 0.924 logistic; Brier is 0.117 vs 0.141; arrival
      MAE is 9.23 s vs 80.26 s zero-change. Group AP gains are +0.067 medium sparse,
      +0.014 nested-switches, and +0.024 tiny topology. The +0.05 per-group gate still
      fails; this repeat uses the same held-out cases and is a seed-sensitivity check,
      not independent generalization evidence.
- [x] Collect fixed first-departure phase alternatives on the medium fixture: 40 matched
      pairs each at 120-second (0 vs 60-second phase) and 600-second (0 vs 120-second
      phase) frequencies. All four cohorts are held out from fitting and their manifest
      checksums are recorded.
- [x] Refit feature-v8 GNN on the 62 prior cohorts plus four phase cohorts, preserving
      the original validation group and all prior test groups. Split: 1,325 train / 20
      validation / 780 heldout episodes; selected epoch 37. Conflict AP is 0.979 vs
      0.954 logistic, but minimum group gain is -0.036, below the +0.05 gate. Artifact:
      `ops-intelligence/models/trained/service-plan-v12-phase-choice-comparable`.
- [x] Evaluate matched phase-choice candidates on the 80 heldout pairs. Pairwise
      objective accuracy is 53.75% at 120-second frequency and 51.25% at 600 seconds.
      Selected mean newly conflicted services are 2.075 vs 2.113 for random choice at
      120 seconds and 3.300 vs 3.313 at 600 seconds; candidate arrival delay is unchanged
      within matched pairs. This does not justify enabling ranking.
- [x] Add 160 training-only OSRD episodes on the tiny topology across four frequency and
      dwell strata (120/240/420/600-second frequency; 60/90/120/180-second stop dwell),
      with randomized first-departure phase. The previously sealed tiny test group remains
      test-only.
- [x] Refit the same 20-group test and original validation split with these new cohorts.
      The 1,485 / 20 / 780 episode fit improves heldout AP to 0.985 vs 0.948 logistic
      and Brier to 0.068 vs 0.126. Tiny-group AP gain is now +0.005 instead of -0.036,
      but minimum group gain remains only +0.002, so the +0.05 gate still fails. Artifact:
      `ops-intelligence/models/trained/service-plan-v12-tiny-topology-expansion`.
- [x] Recheck phase choice with the expanded model: 53.75% pairwise objective accuracy
      at both frequency strata; selected mean new-conflict counts remain 2.075 vs 2.113
      and 3.275 vs 3.313 for random choice. This is not sufficient planning benefit.
- [x] Add two 40-episode training cohorts on the alternate TC1 platform route at 120- and
      600-second frequency. This fills a route-stratum gap while leaving the matched
      route-choice and phase holdouts unchanged.
- [x] Refit the 1,565 / 20 / 780 split on 72 cohorts. Aggregate conflict AP is 0.987 vs
      0.948 logistic and Brier is 0.061 vs 0.121. The minimum group AP gain is +0.003,
      still below the +0.05 safety gate. Artifact:
      `ops-intelligence/models/trained/service-plan-v12-route-topology-expansion`.
- [x] Re-evaluate phase choices. Accuracy is 56.25% for 120-second frequency and 51.25%
      for 600 seconds; average conflicts and arrival delay remain close to random/oracle.
- [x] Acquire and verify 15 monthly Swiss Actual v2 archives for 2025-07 through 2026-09;
      filter `PRODUKT_ID=Zug`, preserve `REAL` status provenance, and build checksummed
      train and held-out profiles. V28 uses the rail-only profile; the earlier mixed-mode
      v25-v27 results are retained as historical records and excluded from this fit.
- [ ] Expand coupled-plan scenarios and independent held-out networks; current v11 has only
      synthetic fixtures and does not establish operator-network transfer. The current
      larger test set still represents only synthetic OSRD infrastructure.
- [ ] Generate route and stop-pattern alternatives from line-level requests, and collect
      paired route-choice and coupled-retiming examples with full-network OSRD outcomes.
- [ ] Add licensed real/operator networks and support stop-pattern/frequency/route-choice
      action data; all checked-in OSRD fixtures are synthetic and do not establish
      operator-network transfer.

### Exploratory 100k-row comparison

These corrected fits use RIDE revision `e5c1ed1126bca1784eb6ea93816dd6cc89afa544`,
100,000 sampled training rows, and a fixed validation seed of 42. The same 5,000-row
early-stop, calibration, and December 2024 evaluation samples are reused across all model
and training seeds. The official 2025 test period was not loaded or scored.

RIDE's tabular labels are named `future_delay_delta_*`; the model predicts these
provider-normalized delta targets, while the shared benchmark table defines absolute
future-delay targets and documents the last-known delay. Seconds metrics therefore invert
the published signed-square-root z-score for the future-delay change,
then add the same row's latest known delay to both the target and model prediction. The
persistence reference uses that latest known delay directly. This corrects an initial
comparison that incorrectly compared the delay change directly with the absolute-delay
reference; those earlier figures are superseded. The RIDE dataset-extension guide says
model-specific builders must preserve the shared Gold-core target definition while changing
only the representation. The pinned Gold Lite schema identifies learned labels as
`future_delay_delta_1` through `future_delay_delta_15`; the training artifact and prediction
CLI now state this output space explicitly. The upstream source implementation of the delta
calculation still needs a direct audit before treating the exact subtraction convention as
fully verified.

| Training seed | Linear MAE (s) | MLP MAE (s) | Persistence MAE (s) |
| ---: | ---: | ---: | ---: |
| 42 | 85.85 | 85.75 | 91.44 |
| 43 | 90.22 | 85.25 | 91.44 |
| 44 | 88.85 | 83.69 | 91.44 |
| Mean ± sample SD | 88.31 ± 2.23 | 84.89 ± 1.07 | 91.44 ± 0.00 |

Macro RMSE averaged 183.10 s for linear and 183.43 s for MLP. The MLP's average MAE
improvement over persistence was 7.16%; linear's was 3.42%. MLP calibrated interval
coverage averaged 80.9%, with about 276 s mean p10–p90 width. This is a small three-seed,
single-period internal comparison, not a deployment claim. The MLP MAE edge is modest and
does not establish generalization across service periods; next validate the target builder,
report horizon and subgroup tails, then do one final untouched 2025 evaluation after the
model and protocol are frozen. Reports and model artifacts remain under ignored
`ops-intelligence/data/rail/ride-forecast-{linear,mlp}-seedNN-100k/` plus the seed-42
directories without the seed suffix.

### 500k-row sample-size comparison

The next fit increased only the fit sample to 500,000 rows. It used training seeds 42,
43, and 44, with validation seed 42 and the same early-stop, calibration, and evaluation
rows for every run. The evaluation-index SHA-256 matched across all six reports, and the
2025 test period remained unused.

| Fit rows | Model | Macro MAE, mean ± sample SD (s) | Macro RMSE, mean ± sample SD (s) | MAE improvement over persistence |
| ---: | --- | ---: | ---: | ---: |
| 100k | Linear | 88.31 ± 2.23 | 183.10 ± 0.07 | 3.42% |
| 100k | MLP | 84.89 ± 1.07 | 183.43 ± 1.33 | 7.16% |
| 500k | Linear | 87.90 ± 3.06 | 183.49 ± 1.11 | 3.87% |
| 500k | MLP | 80.73 ± 0.63 | 180.01 ± 0.71 | 11.71% |

The persistence macro MAE was 91.44 s on the shared evaluation rows. Increasing to 500k
helped the MLP by about 4.16 s (4.9%) in mean MAE, while the linear baseline was nearly
unchanged. At 500k the MLP's mean calibrated coverage was 80.7%, with about 262 s mean
p10–p90 width. This supports using the MLP as the current offline candidate, not a serving
decision: it still needs the upstream target-builder audit, wider temporal and subgroup
evaluation, and one final frozen evaluation on the reserved 2025 test period. The
500k reports and artifacts are under ignored
`ops-intelligence/data/rail/ride-forecast-{linear,mlp}-seed{42,43,44}-500k/`.

## References

- RIDE dataset card and pinned model-ready data: <https://huggingface.co/datasets/orailix/ride-gold-lite>
- RIDE benchmark paper: <https://arxiv.org/abs/2606.05070>
- RIDE source and construction code: <https://github.com/orailix/ride>
- RIDE model-specific Gold dataset extension guide: <https://github.com/orailix/ride/blob/main/docs/create_gold_dataset.md>
- RIDE Gold Lite file manifest (includes GNN schema/shard URLs and checksums): <https://huggingface.co/datasets/orailix/ride-gold-lite/blob/main/ride_gold_lite_croissant.json>
- Third-party RIDE representative sample (schema smoke only; verify license before use): <https://huggingface.co/datasets/ano6060/anonymous-ride-sample>
- Infrabel raw monthly punctuality: <https://opendata.infrabel.be/explore/dataset/stiptheid-gegevens-maandelijksebestanden/>
- Infrabel freight punctuality: <https://opendata.infrabel.be/explore/dataset/stiptheid-van-de-goederentreinen/>
- Swiss Actual data v2: <https://data.opentransportdata.swiss/en/dataset/ist-daten-v2>
- Swiss public transport GTFS documentation: <https://opentransportdata.swiss/en/cookbook/timetable-cookbook/gtfs/>
- Swiss open data terms: <https://opentransportdata.swiss/en/terms-of-use/>
