"""Advisory operations-intelligence HTTP service."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import UUID

import uvicorn
from fastapi import FastAPI, Response, status
from fastapi.responses import HTMLResponse, JSONResponse

from osrd_ops_intelligence import __version__
from osrd_ops_intelligence.api.contracts import (
    CompleteTimetablePlanRequest,
    CompleteTimetablePlanResponse,
    ExistingServiceRetimingPlanRequest,
    ExistingServiceRetimingPlanResponse,
    HourlyServicePlanResponse,
    HourlyServiceRequest,
    InsertionRankingRequest,
    InsertionRankingResponse,
    PlanAuditRecord,
    PlanIntentRequest,
    PlanIntentResponse,
    PlanningRequest,
    PlanningResponse,
    PlanProblem,
    ServicePlanRequest,
    ServicePlanResponse,
    ServicePlanReviewReceipt,
    ServicePlanReviewSubmission,
    ServiceRequest,
    ServiceRequestValidationResponse,
    SnapshotBinding,
)
from osrd_ops_intelligence.service.insertion_ranking import (
    InsertionRankingUnavailable,
    rank_insertion_candidates,
)
from osrd_ops_intelligence.service.osrd_client import (
    EditoastError,
    EditoastOSRDClient,
    OSRDClient,
    UnsupportedSnapshot,
)
from osrd_ops_intelligence.service.plan_audit import (
    AuditConflict,
    AuditStoreError,
    PlanAuditStore,
)
from osrd_ops_intelligence.service.plan_intent import (
    PlanIntentUnavailable,
    interpret_plan_intent,
)
from osrd_ops_intelligence.service.planning import PlanningInputError, plan
from osrd_ops_intelligence.service.service_requests import (
    plan_complete_timetable,
    plan_existing_service_retimings,
    plan_hourly_service_request,
    plan_service_request,
    validate_service_request,
)


def _client_from_environment() -> OSRDClient | None:
    base_url = os.getenv("OSRD_EDITOAST_URL")
    if not base_url:
        return None
    headers: dict[str, str] = {}
    authorization = os.getenv("OSRD_AUTHORIZATION")
    if authorization:
        headers["Authorization"] = authorization
    return EditoastOSRDClient(
        base_url,
        headers=headers,
    )


def _problem(
    code: str,
    message: str,
    request: PlanningRequest | ServiceRequest | HourlyServiceRequest | ExistingServiceRetimingPlanRequest,
    status_code: int,
) -> JSONResponse:
    payload = PlanProblem(
        code=code,
        message=message,
        request_id=request.request_id,
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def create_app(osrd_client: OSRDClient | None = None) -> FastAPI:
    """Build the API with an optional OSRD adapter; absent configuration fails closed."""

    configured_client = osrd_client if osrd_client is not None else _client_from_environment()
    blind_review_mode = os.getenv("OSRD_OPS_BLIND_REVIEW_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    review_infrastructure_group = os.getenv("OSRD_OPS_REVIEW_INFRASTRUCTURE_GROUP", "").strip()
    if blind_review_mode and not review_infrastructure_group:
        raise ValueError(
            "OSRD_OPS_REVIEW_INFRASTRUCTURE_GROUP is required for blind review collection"
        )
    insertion_model_path = (
        None if blind_review_mode else os.getenv("OSRD_OPS_INSERTION_MODEL")
    )
    audit_store = PlanAuditStore(os.getenv("OSRD_OPS_AUDIT_DB", "data/plan-audit.sqlite3"))
    application = FastAPI(
        title="OSRD Operations Intelligence",
        version=__version__,
        description=(
            "Advisory OSRD timetable planning and delay-recovery service. It validates "
            "the unchanged baseline and candidate timetables through OSRD pathfinding, "
            "simulation, and conflict detection; it can rank outcome-free service-insertion "
            "alternatives when an offline-eligible model is configured, and validate "
            "bounded frequency-service requests. API planning routes do not mutate "
            "timetables; the planner page can create a separate draft scenario after a "
            "planner selects a validated option."
        ),
    )

    @application.get("/healthz", tags=["operations"])
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "osrd-ops-intelligence", "version": __version__}

    @application.get(
        "/readyz",
        tags=["operations"],
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "description": "Planning is not configured or the OSRD backend is unreachable."
            }
        },
    )
    async def readiness(response: Response) -> dict[str, bool | str | None]:
        planning_configured = configured_client is not None
        readiness_check = getattr(configured_client, "check_ready", None)
        backend_reachable = None
        if planning_configured and readiness_check is not None:
            backend_reachable = await readiness_check()
        if backend_reachable is not True:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "ready" if backend_reachable is True else "not_ready",
            "planning_configured": planning_configured,
            "osrd_backend_reachable": backend_reachable,
        }

    @application.get("/planner", response_class=HTMLResponse, include_in_schema=False)
    def planner_page() -> HTMLResponse:
        page = Path(__file__).with_name("planner.html").read_text(encoding="utf-8")
        page = page.replace("__BLIND_REVIEW_MODE__", "true" if blind_review_mode else "false")
        page = page.replace(
            "__REVIEW_INFRASTRUCTURE_GROUP__",
            json.dumps(review_infrastructure_group or "").replace("<", "\\u003c"),
        )
        if blind_review_mode:
            page = page.replace(
                '<h1>Plan a service in OSRD</h1>',
                '<h1>Plan a service in OSRD</h1><p class="blind-review-banner" role="status">Blind planner review is active. Candidate ordering uses the configured deterministic policy; research model scores and suggestions are disabled for this session.</p>',
            )
        return HTMLResponse(page)

    @application.get("/locales/{language}", response_class=JSONResponse, include_in_schema=False)
    def planner_locale(language: str) -> JSONResponse:
        if language not in {"de", "en", "fr", "pt"}:
            return JSONResponse({"detail": "Unsupported planner language."}, status_code=404)
        locale_path = Path(__file__).with_name("locales") / f"{language}.json"
        return JSONResponse(json.loads(locale_path.read_text(encoding="utf-8")))

    @application.post(
        "/v1/operations/plan-intent",
        response_model=PlanIntentResponse | PlanProblem,
        responses={
            status.HTTP_400_BAD_REQUEST: {"model": PlanProblem},
            status.HTTP_502_BAD_GATEWAY: {"model": PlanProblem},
            status.HTTP_503_SERVICE_UNAVAILABLE: {"model": PlanProblem},
        },
        tags=["planning"],
        summary="Interpret a natural-language planning request as a reviewable draft",
        description=(
            "Sends only the request text to the configured AI provider after explicit per-request "
            "consent. It does not resolve OSRD identifiers, create plans, or change timetables."
        ),
    )
    async def interpret_plan_request(request: PlanIntentRequest) -> Any:
        try:
            return PlanIntentResponse(
                draft=await interpret_plan_intent(request)
            )
        except PlanIntentUnavailable as error:
            is_consent_error = not request.external_processing_confirmed
            return JSONResponse(
                status_code=(
                    status.HTTP_400_BAD_REQUEST
                    if is_consent_error
                    else status.HTTP_503_SERVICE_UNAVAILABLE
                ),
                content=PlanProblem(
                    code=(
                        "external_processing_not_confirmed"
                        if is_consent_error
                        else "plan_intent_unavailable"
                    ),
                    message=str(error),
                ).model_dump(mode="json"),
            )

    @application.get(
        "/v1/operations/snapshots/{infra_id}/{timetable_id}",
        response_model=SnapshotBinding | PlanProblem,
        responses={
            status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": PlanProblem},
            status.HTTP_501_NOT_IMPLEMENTED: {"model": PlanProblem},
            status.HTTP_502_BAD_GATEWAY: {"model": PlanProblem},
        },
        tags=["planning"],
        summary="Read current OSRD version bindings for a timetable snapshot",
    )
    async def describe_snapshot(infra_id: str, timetable_id: str) -> Any:
        if configured_client is None:
            payload = PlanProblem(
                code="planning_not_configured",
                message="Set OSRD_EDITOAST_URL to read OSRD snapshot bindings.",
            )
            return JSONResponse(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                content=payload.model_dump(mode="json"),
            )
        try:
            snapshot = await configured_client.describe_snapshot(
                infra_id=infra_id,
                timetable_id=timetable_id,
            )
            return SnapshotBinding.model_validate(asdict(snapshot))
        except UnsupportedSnapshot as error:
            payload = PlanProblem(code="snapshot_unavailable", message=str(error))
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content=payload.model_dump(mode="json"),
            )
        except EditoastError as error:
            payload = PlanProblem(code="osrd_unavailable", message=str(error))
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content=payload.model_dump(mode="json"),
            )

    @application.post(
        "/v1/operations/plans",
        response_model=PlanningResponse | PlanProblem,
        responses={
            status.HTTP_409_CONFLICT: {"model": PlanProblem},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": PlanProblem},
            status.HTTP_501_NOT_IMPLEMENTED: {"model": PlanProblem},
            status.HTTP_502_BAD_GATEWAY: {"model": PlanProblem},
            status.HTTP_503_SERVICE_UNAVAILABLE: {"model": PlanProblem},
        },
        tags=["planning"],
        summary="Generate and OSRD-validate delay-recovery candidates",
    )
    async def create_plan(request: PlanningRequest) -> Any:
        try:
            previous = audit_store.replay(request)
            if previous is not None:
                return previous.result
        except AuditConflict as error:
            return _problem("request_id_conflict", str(error), request, status.HTTP_409_CONFLICT)
        except AuditStoreError as error:
            return _problem(
                "audit_store_unavailable", str(error), request, status.HTTP_503_SERVICE_UNAVAILABLE
            )

        if configured_client is None:
            return _problem(
                "planning_not_configured",
                "Set OSRD_EDITOAST_URL to enable isolated OSRD candidate validation.",
                request,
                status.HTTP_501_NOT_IMPLEMENTED,
            )
        try:
            result = await plan(request, configured_client)
            return audit_store.record(request, result).result
        except PlanningInputError as error:
            return _problem(
                "planning_inputs_incomplete",
                str(error),
                request,
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )
        except UnsupportedSnapshot as error:
            return _problem(
                "snapshot_binding_failed", str(error), request, status.HTTP_409_CONFLICT
            )
        except EditoastError as error:
            return _problem(
                "osrd_validation_failed", str(error), request, status.HTTP_502_BAD_GATEWAY
            )
        except ValueError as error:
            return _problem(
                "planning_inputs_invalid",
                str(error),
                request,
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )
        except AuditConflict as error:
            return _problem("request_id_conflict", str(error), request, status.HTTP_409_CONFLICT)
        except AuditStoreError as error:
            return _problem(
                "audit_store_unavailable", str(error), request, status.HTTP_503_SERVICE_UNAVAILABLE
            )

    @application.post(
        "/v1/operations/insertion-rankings",
        response_model=InsertionRankingResponse | PlanProblem,
        responses={
            status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": PlanProblem},
            status.HTTP_501_NOT_IMPLEMENTED: {"model": PlanProblem},
            status.HTTP_503_SERVICE_UNAVAILABLE: {"model": PlanProblem},
        },
        tags=["planning"],
        summary="Rank outcome-free single-service insertion alternatives",
        description=(
            "Returns an advisory ranking only. Configure an artifact that passed "
            "its offline ranking gate with OSRD_OPS_INSERTION_MODEL. Every alternative "
            "must share the same baseline timetable and background services. This route "
            "does not generate or validate timetables; callers must run complete "
            "candidates through OSRD pathfinding, simulation, and conflict detection "
            "before presenting them as feasible."
        ),
    )
    async def rank_insertions(request: InsertionRankingRequest) -> Any:
        if not insertion_model_path:
            payload = PlanProblem(
                code="insertion_model_not_configured",
                message="Set OSRD_OPS_INSERTION_MODEL to an offline-eligible model artifact.",
                request_id=request.request_id,
            )
            return JSONResponse(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                content=payload.model_dump(mode="json"),
            )
        try:
            return await asyncio.to_thread(
                rank_insertion_candidates,
                request,
                model_path=Path(insertion_model_path),
            )
        except InsertionRankingUnavailable as error:
            payload = PlanProblem(
                code="insertion_model_not_eligible",
                message=str(error),
                request_id=request.request_id,
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=payload.model_dump(mode="json"),
            )
        except OSError as error:
            payload = PlanProblem(
                code="insertion_model_unavailable",
                message=str(error),
                request_id=request.request_id,
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=payload.model_dump(mode="json"),
            )
        except (KeyError, TypeError, ValueError) as error:
            payload = PlanProblem(
                code="insertion_ranking_inputs_invalid",
                message=str(error),
                request_id=request.request_id,
            )
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content=payload.model_dump(mode="json"),
            )

    @application.post(
        "/v1/operations/service-requests/validate",
        response_model=ServiceRequestValidationResponse | PlanProblem,
        responses={
            status.HTTP_409_CONFLICT: {"model": PlanProblem},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": PlanProblem},
            status.HTTP_501_NOT_IMPLEMENTED: {"model": PlanProblem},
            status.HTTP_502_BAD_GATEWAY: {"model": PlanProblem},
        },
        tags=["planning"],
        summary="Build and OSRD-validate a bounded frequency service request",
        description=(
            "Expands an explicit path, stop pattern, UTC operating window, and frequency "
            "into individual calendar departures. The complete source timetable plus "
            "those departures is validated in an isolated temporary timetable. This "
            "does not search for a route or retime existing services."
        ),
    )
    async def validate_service(request: ServiceRequest) -> Any:
        if configured_client is None:
            return _problem(
                "planning_not_configured",
                "Set OSRD_EDITOAST_URL to enable isolated OSRD candidate validation.",
                request,
                status.HTTP_501_NOT_IMPLEMENTED,
            )
        try:
            return await validate_service_request(request, configured_client)
        except UnsupportedSnapshot as error:
            return _problem(
                "snapshot_binding_failed", str(error), request, status.HTTP_409_CONFLICT
            )
        except EditoastError as error:
            return _problem(
                "osrd_validation_failed", str(error), request, status.HTTP_502_BAD_GATEWAY
            )
        except (KeyError, TypeError, ValueError) as error:
            return _problem(
                "service_request_invalid",
                str(error),
                request,
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )

    @application.post(
        "/v1/operations/service-requests/plans",
        response_model=ServicePlanResponse | PlanProblem,
        responses={
            status.HTTP_409_CONFLICT: {"model": PlanProblem},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": PlanProblem},
            status.HTTP_501_NOT_IMPLEMENTED: {"model": PlanProblem},
            status.HTTP_502_BAD_GATEWAY: {"model": PlanProblem},
        },
        tags=["planning"],
        summary="Generate bounded service and retiming plans validated by OSRD",
        description=(
            "Retimes only explicitly eligible existing train schedules within configured "
            "bounds, explores bounded OSRD route alternatives and first-departure phases, "
            "inserts the requested "
            "frequency service, and validates every complete "
            "candidate through OSRD. Candidate generation is bounded and does not guarantee "
            "a globally optimal plan. An offline-eligible insertion model can add a separate "
            "research suggestion for eligible single-departure, direct-path options; it does "
            "not alter deterministic ordering, feasibility, or the planner's choice."
        ),
    )
    async def plan_service(request: ServicePlanRequest) -> Any:
        if configured_client is None:
            return _problem(
                "planning_not_configured",
                "Set OSRD_EDITOAST_URL to enable isolated OSRD candidate validation.",
                request.service,
                status.HTTP_501_NOT_IMPLEMENTED,
            )
        try:
            return await plan_service_request(
                request,
                configured_client,
                insertion_model_path=(Path(insertion_model_path) if insertion_model_path else None),
            )
        except UnsupportedSnapshot as error:
            return _problem(
                "snapshot_binding_failed", str(error), request.service, status.HTTP_409_CONFLICT
            )
        except EditoastError as error:
            return _problem(
                "osrd_validation_failed", str(error), request.service, status.HTTP_502_BAD_GATEWAY
            )
        except (KeyError, TypeError, ValueError) as error:
            return _problem(
                "service_plan_invalid",
                str(error),
                request.service,
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )

    @application.post(
        "/v1/operations/complete-timetable/plans",
        response_model=CompleteTimetablePlanResponse | PlanProblem,
        responses={
            status.HTTP_409_CONFLICT: {"model": PlanProblem},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": PlanProblem},
            status.HTTP_501_NOT_IMPLEMENTED: {"model": PlanProblem},
            status.HTTP_502_BAD_GATEWAY: {"model": PlanProblem},
        },
        tags=["planning"],
        summary="Generate and jointly validate a bounded multi-service timetable",
        description=(
            "Combines every requested calendar service pattern and bounded edits to explicitly "
            "eligible existing services into each candidate. OSRD validates the complete "
            "timetable with pathfinding, simulation, and conflict detection. Candidate search "
            "is bounded; OSRD validation remains authoritative."
        ),
    )
    async def plan_complete(request: CompleteTimetablePlanRequest) -> Any:
        service = request.services[0]
        if configured_client is None:
            return _problem(
                "planning_not_configured",
                "Set OSRD_EDITOAST_URL to enable isolated OSRD candidate validation.",
                service,
                status.HTTP_501_NOT_IMPLEMENTED,
            )
        try:
            return await plan_complete_timetable(request, configured_client)
        except UnsupportedSnapshot as error:
            return _problem(
                "snapshot_binding_failed", str(error), service, status.HTTP_409_CONFLICT
            )
        except EditoastError as error:
            return _problem(
                "osrd_validation_failed", str(error), service, status.HTTP_502_BAD_GATEWAY
            )
        except (KeyError, TypeError, ValueError) as error:
            return _problem(
                "complete_timetable_plan_invalid",
                str(error),
                service,
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )

    @application.post(
        "/v1/operations/existing-service-retimings/plans",
        response_model=ExistingServiceRetimingPlanResponse | PlanProblem,
        responses={
            status.HTTP_409_CONFLICT: {"model": PlanProblem},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": PlanProblem},
            status.HTTP_502_BAD_GATEWAY: {"model": PlanProblem},
        },
        tags=["planning"],
        summary="Generate and validate retiming-only timetable options",
        description=(
            "Searches bounded departure-time changes to explicitly eligible existing services. "
            "Every option is validated against the full source timetable by OSRD; no service "
            "is inserted and the source timetable is not changed."
        ),
    )
    async def plan_existing_retimings(request: ExistingServiceRetimingPlanRequest) -> Any:
        if configured_client is None:
            return _problem(
                "planning_not_configured",
                "Set OSRD_EDITOAST_URL to enable isolated OSRD candidate validation.",
                request,
                status.HTTP_501_NOT_IMPLEMENTED,
            )
        try:
            return await plan_existing_service_retimings(request, configured_client)
        except UnsupportedSnapshot as error:
            return _problem("snapshot_binding_failed", str(error), request, status.HTTP_409_CONFLICT)
        except EditoastError as error:
            return _problem("osrd_validation_failed", str(error), request, status.HTTP_502_BAD_GATEWAY)
        except (KeyError, TypeError, ValueError) as error:
            return _problem(
                "existing_service_retiming_invalid",
                str(error),
                request,
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )

    @application.post(
        "/v1/operations/hourly-service-requests/plans",
        response_model=HourlyServicePlanResponse | PlanProblem,
        responses={
            status.HTTP_409_CONFLICT: {"model": PlanProblem},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": PlanProblem},
            status.HTTP_501_NOT_IMPLEMENTED: {"model": PlanProblem},
            status.HTTP_502_BAD_GATEWAY: {"model": PlanProblem},
        },
        tags=["planning"],
        summary="Generate recurring hourly service options validated by OSRD",
        description=(
            "Explores bounded cycle-relative departure offsets for an OSRD HOURLY timetable. "
            "Each option is a single paced schedule validated with the complete source "
            "timetable through pathfinding, simulation, and conflict detection. Responses "
            "report offsets, not calendar timestamps; this route does not retime existing trains."
        ),
    )
    async def plan_hourly_service(request: HourlyServiceRequest) -> Any:
        if configured_client is None:
            return _problem(
                "planning_not_configured",
                "Set OSRD_EDITOAST_URL to enable isolated OSRD candidate validation.",
                request,
                status.HTTP_501_NOT_IMPLEMENTED,
            )
        try:
            return await plan_hourly_service_request(request, configured_client)
        except UnsupportedSnapshot as error:
            return _problem(
                "snapshot_binding_failed", str(error), request, status.HTTP_409_CONFLICT
            )
        except EditoastError as error:
            return _problem(
                "osrd_validation_failed", str(error), request, status.HTTP_502_BAD_GATEWAY
            )
        except (KeyError, TypeError, ValueError) as error:
            return _problem(
                "hourly_service_plan_invalid",
                str(error),
                request,
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            )

    @application.get(
        "/v1/operations/plans/{request_id}",
        response_model=PlanAuditRecord | PlanProblem,
        responses={
            status.HTTP_404_NOT_FOUND: {"model": PlanProblem},
            status.HTTP_503_SERVICE_UNAVAILABLE: {"model": PlanProblem},
        },
        tags=["planning"],
        summary="Retrieve an immutable completed planning record",
    )
    async def get_plan_record(request_id: UUID) -> Any:
        try:
            record = audit_store.get(request_id)
        except AuditStoreError as error:
            payload = PlanProblem(code="audit_store_unavailable", message=str(error))
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=payload.model_dump(mode="json"),
            )
        if record is None:
            payload = PlanProblem(
                code="plan_not_found", message="No completed plan has this request_id."
            )
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=payload.model_dump(mode="json"),
            )
        return record

    @application.post(
        "/v1/operations/service-plan-reviews",
        response_model=ServicePlanReviewReceipt | PlanProblem,
        responses={
            status.HTTP_409_CONFLICT: {"model": PlanProblem},
            status.HTTP_503_SERVICE_UNAVAILABLE: {"model": PlanProblem},
        },
        tags=["planning"],
        summary="Save a planner-submitted service-plan review record",
        description=(
            "Stores an immutable review record only when the planner explicitly submits it. "
            "Records support later preference research; a planner's choice is not evidence "
            "that an option was operationally accepted or improved outcomes."
        ),
    )
    async def save_service_plan_review(
        submission: ServicePlanReviewSubmission,
    ) -> Any:
        if (
            not blind_review_mode
            and submission.record.get("review_protocol") == "blind_planner_choice_v1"
        ):
            payload = PlanProblem(
                code="blind_review_mode_inactive",
                message="Blind review records can only be submitted to a blind review session.",
            )
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content=payload.model_dump(mode="json"),
            )
        if (
            blind_review_mode
            and submission.record.get("review_protocol") != "blind_planner_choice_v1"
        ):
            payload = PlanProblem(
                code="blind_review_protocol_required",
                message="This planner is collecting blind review data; submit a record created in this session.",
            )
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content=payload.model_dump(mode="json"),
            )
        if (
            blind_review_mode
            and submission.record.get("study_infrastructure_group")
            != review_infrastructure_group
        ):
            payload = PlanProblem(
                code="blind_review_infrastructure_group_mismatch",
                message="Review infrastructure group must match this collection session.",
            )
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content=payload.model_dump(mode="json"),
            )
        try:
            return audit_store.record_service_plan_review(submission)
        except AuditConflict as error:
            payload = PlanProblem(code="review_id_conflict", message=str(error))
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content=payload.model_dump(mode="json"),
            )
        except AuditStoreError as error:
            payload = PlanProblem(code="audit_store_unavailable", message=str(error))
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=payload.model_dump(mode="json"),
            )

    return application


app = create_app()


def run() -> None:
    uvicorn.run("osrd_ops_intelligence.service.app:app", host="0.0.0.0", port=8095)
