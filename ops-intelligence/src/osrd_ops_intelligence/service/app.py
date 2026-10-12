"""Advisory operations-intelligence HTTP service."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

import uvicorn
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from osrd_ops_intelligence import __version__
from osrd_ops_intelligence.api.contracts import (
    PlanningRequest,
    PlanningResponse,
    PlanProblem,
    SnapshotBinding,
)
from osrd_ops_intelligence.service.osrd_client import (
    EditoastError,
    EditoastOSRDClient,
    OSRDClient,
    UnsupportedSnapshot,
)
from osrd_ops_intelligence.service.planning import PlanningInputError, plan


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
    code: str, message: str, request: PlanningRequest, status_code: int
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
    application = FastAPI(
        title="OSRD Operations Intelligence",
        version=__version__,
        description=(
            "Advisory timetable candidate service. It only returns candidates that "
            "pass OSRD simulation and conflict checks; it never applies a plan."
        ),
    )

    @application.get("/healthz", tags=["operations"])
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "osrd-ops-intelligence", "version": __version__}

    @application.get(
        "/v1/operations/snapshots/{infra_id}/{timetable_id}",
        response_model=SnapshotBinding | PlanProblem,
        responses={
            status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": PlanProblem},
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
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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
            status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": PlanProblem},
            status.HTTP_501_NOT_IMPLEMENTED: {"model": PlanProblem},
            status.HTTP_502_BAD_GATEWAY: {"model": PlanProblem},
        },
        tags=["planning"],
        summary="Generate and OSRD-validate delay-recovery candidates",
    )
    async def create_plan(request: PlanningRequest) -> Any:
        if configured_client is None:
            return _problem(
                "planning_not_configured",
                "Set OSRD_EDITOAST_URL to enable isolated OSRD candidate validation.",
                request,
                status.HTTP_501_NOT_IMPLEMENTED,
            )
        try:
            return await plan(request, configured_client)
        except PlanningInputError as error:
            return _problem(
                "planning_inputs_incomplete",
                str(error),
                request,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
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
                "planning_inputs_invalid", str(error), request, status.HTTP_422_UNPROCESSABLE_ENTITY
            )

    return application


app = create_app()


def run() -> None:
    uvicorn.run("osrd_ops_intelligence.service.app:app", host="0.0.0.0", port=8095)
