"""Fail-closed advisory ranking for outcome-free service-insertion candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from osrd_ops_intelligence.api.contracts import (
    InsertionCandidateScore,
    InsertionRankingRequest,
    InsertionRankingResponse,
    InsertionServiceImpact,
)


class InsertionRankingUnavailable(RuntimeError):
    """Raised when no configured model has passed the offline ranking gate."""


def _baseline_context_digest(episode: dict[str, Any]) -> str:
    inputs = episode["inputs"]
    baseline = episode["outputs"]["paired"]["baseline"]
    context = {
        "snapshot": episode.get("snapshot"),
        "infra_id": inputs["scenario"].get("infra_id"),
        "baseline_schedule_payloads": inputs["baseline_schedule_payloads"],
        "baseline_schedule_ids": baseline["schedule_ids"],
        "baseline_simulation_summaries": baseline["simulation_summaries"],
    }
    canonical = json.dumps(context, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _validate_candidate_episode(episode: dict[str, Any]) -> tuple[list[int], int]:
    if episode.get("schema") != "osrd-ops-replay-episode/v2":
        raise ValueError("Candidate replay has an unsupported episode schema")
    if "labels" in episode:
        raise ValueError("Candidate replay must not contain training labels")
    snapshot = episode.get("snapshot")
    if isinstance(snapshot, dict) and any(key.startswith("treatment_") for key in snapshot):
        raise ValueError("Candidate replay must not contain treatment timetable identifiers")
    inputs = episode.get("inputs")
    outputs = episode.get("outputs")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise ValueError("Candidate replay must contain inputs and baseline outputs")
    if set(outputs) != {"paired"}:
        raise ValueError("Candidate replay may contain only the paired baseline output")
    paired = outputs.get("paired")
    if not isinstance(paired, dict) or set(paired) != {"baseline"}:
        raise ValueError("Candidate replay must not contain treatment outputs")
    baseline = paired["baseline"]
    if not isinstance(baseline, dict):
        raise ValueError("Candidate baseline output has an invalid shape")
    ids = baseline.get("schedule_ids")
    baseline_payloads = inputs.get("baseline_schedule_payloads")
    candidate_payloads = inputs.get("schedule_payloads")
    paths = inputs.get("resource_paths_by_train")
    action = inputs.get("action")
    if not isinstance(ids, list) or not all(isinstance(value, int) for value in ids):
        raise ValueError("Candidate baseline must list integer train-schedule IDs")
    if not isinstance(baseline_payloads, list) or len(ids) != len(baseline_payloads):
        raise ValueError("Candidate baseline IDs and schedule payloads do not match")
    if not ids:
        raise ValueError("Service-insertion ranking requires at least one existing service")
    if not isinstance(candidate_payloads, list) or len(candidate_payloads) != len(ids) + 1:
        raise ValueError("Each candidate must append exactly one new service to the baseline")
    if candidate_payloads[: len(ids)] != baseline_payloads:
        raise ValueError("Candidate background services must match the baseline timetable")
    if not isinstance(paths, list) or len(paths) != len(candidate_payloads):
        raise ValueError("Candidate must include one pre-simulation resource path per service")
    if not all(isinstance(path, dict) and path.get("track_section_ranges") for path in paths):
        raise ValueError("Every candidate service needs an OSRD track-resource path")
    if not isinstance(action, dict) or action.get("type") != "service-insertion":
        raise ValueError("Candidate action must be a service insertion")
    if action.get("train_index") != len(ids):
        raise ValueError("The inserted service must be the final schedule in the candidate")
    if not isinstance(baseline.get("simulation_summaries"), dict):
        raise ValueError("Candidate baseline must include OSRD simulation summaries")
    return ids, len(ids)


def rank_insertion_candidates(
    request: InsertionRankingRequest,
    *,
    model_path: Path,
) -> InsertionRankingResponse:
    """Rank alternatives against an identical baseline; never certify feasibility."""
    from osrd_ops_intelligence.training.predict_replay_gnn import predict_episode

    validated: list[tuple[str, dict[str, Any], list[int], int]] = []
    baseline_digest: str | None = None
    treatment_background_digest: str | None = None
    for candidate in request.candidates:
        episode = candidate.replay_episode
        baseline_ids, baseline_count = _validate_candidate_episode(episode)
        digest = _baseline_context_digest(episode)
        if baseline_digest is None:
            baseline_digest = digest
        elif digest != baseline_digest:
            raise ValueError("All alternatives must use the same baseline timetable context")
        candidate_context = {
            "background_schedule_payloads": episode["inputs"]["schedule_payloads"][:baseline_count],
            "resource_paths": episode["inputs"]["resource_paths_by_train"][:baseline_count],
            "headway_s": episode["inputs"]["scenario"].get("headway_s"),
            "rolling_stock": episode["inputs"]["scenario"].get("rolling_stock"),
            "background_perturbations": episode["inputs"].get("background_perturbations"),
        }
        candidate_digest = hashlib.sha256(
            json.dumps(candidate_context, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if treatment_background_digest is None:
            treatment_background_digest = candidate_digest
        elif candidate_digest != treatment_background_digest:
            raise ValueError("Alternatives must preserve the same background service inputs")
        validated.append((candidate.candidate_id, episode, baseline_ids, baseline_count))

    scores: list[tuple[str, float, float, list[InsertionServiceImpact]]] = []
    model_digest: str | None = None
    for candidate_id, episode, baseline_ids, baseline_count in validated:
        prediction = predict_episode(model_path, episode)
        if not prediction["offline_candidate_ranking_eligible"]:
            raise InsertionRankingUnavailable(
                "Configured insertion model did not pass its offline candidate-ranking gate"
            )
        digest = str(prediction["model_sha256"])
        if model_digest is None:
            model_digest = digest
        elif digest != model_digest:
            raise InsertionRankingUnavailable("Model artifact changed while ranking alternatives")
        service_rows = prediction["services"][:baseline_count]
        if len(service_rows) != baseline_count:
            raise ValueError("Model output does not match the baseline timetable services")
        probabilities = [float(row["conflict_probability"]) for row in service_rows]
        impacts = [
            InsertionServiceImpact(
                baseline_schedule_id=baseline_ids[index],
                conflict_probability=probabilities[index],
                conflict_risk_at_threshold=service_rows[index]["conflict_risk_at_threshold"],
            )
            for index in range(baseline_count)
        ]
        scores.append(
            (candidate_id, sum(probabilities) / len(probabilities), max(probabilities), impacts)
        )

    scores.sort(key=lambda row: (row[1], row[2], row[0]))
    return InsertionRankingResponse(
        request_id=request.request_id,
        model_sha256=model_digest or "",
        offline_candidate_ranking_eligible=True,
        generated_candidate_count=len(scores),
        ranked_candidates=[
            InsertionCandidateScore(
                rank=rank,
                candidate_id=candidate_id,
                mean_existing_service_conflict_probability=mean_probability,
                max_existing_service_conflict_probability=max_probability,
                affected_services=impacts,
            )
            for rank, (candidate_id, mean_probability, max_probability, impacts) in enumerate(
                scores, start=1
            )
        ],
    )
