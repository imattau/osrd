"""Small HTTP shell for the operations-intelligence service."""

import uvicorn
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from osrd_ops_intelligence import __version__
from osrd_ops_intelligence.contracts import PlanProblem, PlanningRequest

app = FastAPI(
    title="OSRD Operations Intelligence",
    version=__version__,
    description=(
        "Advisory planning API scaffold. It does not yet generate or apply plans; "
        "OSRD remains the feasibility authority."
    ),
)


@app.get("/healthz", tags=["operations"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "osrd-ops-intelligence", "version": __version__}


@app.post(
    "/v1/operations/plans",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    response_model=PlanProblem,
    tags=["planning"],
    summary="Submit an operations-planning request",
)
def create_plan(request: PlanningRequest) -> JSONResponse:
    """Publish the request contract; planning is not wired up yet."""

    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content=PlanProblem(
            code="planning_not_configured",
            message=(
                "The planning orchestrator is not implemented. No plan was generated "
                "or applied."
            ),
            request_id=request.request_id,
        ).model_dump(mode="json"),
    )


def run() -> None:
    uvicorn.run("osrd_ops_intelligence.api:app", host="0.0.0.0", port=8095)
