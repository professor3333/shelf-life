"""The service. `POST /predict`, `GET /health`, and no feature logic anywhere.

    request  ->  pydantic  ->  contract  ->  the artifact's Pipeline  ->  JSON

Read that chain for what is *missing* from it. There is no imputation here, no
encoding, no derived column, no threshold arithmetic. Every one of those lives
inside the object `src/models/freeze.py` wrote, and this module's job is to hand
it a validated payload and serialise what comes back. That is the requirement in
one sentence: **the serving path uses the same fitted pipeline object as
training, not a re-implementation of it.** A re-implementation agrees on the day
it is written and drifts quietly afterwards, which is training/serving skew —
leakage's production-shaped sibling.

**The artifact is loaded once, at startup**, into `app.state`. Not per request:
unpickling a booster and its transformer is the expensive part, and doing it
inside the handler would make the first slow response indistinguishable from a
slow model.

**A missing artifact is a state, not a crash.** The process starts, `/health`
answers 200 and says `model_loaded: false`, and `/predict` refuses with 503. The
alternative — refusing to boot — turns "the artifact was not built" into a
container that will not start, and the logs are the same either way but the
diagnosis is much slower. This matters here more than usual, because the panel
is still too shallow to freeze a real model against.

**Bad input is a 4xx, always.** Pydantic answers the shape; `InvalidPayload`
from the contract answers the content; both come back as 422 with a body that
says which field and why. A 500 means *this service* is broken, and spending
that signal on a caller's typo wastes it.

Run it::

    uvicorn api.main:app --reload
    open http://localhost:8000/docs
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.schemas import HealthResponse, PostingRequest, PredictionResponse
from src.inference.artifact import DEFAULT_ARTIFACT, ArtifactError
from src.inference.contract import InvalidPayload, describe
from src.inference.predict import Predictor

#: Where the artifact lives, overridable so a container can mount one elsewhere
#: and a test can point at one it built itself.
ARTIFACT_ENV = "SHELF_LIFE_ARTIFACT"

DESCRIPTION = """
Predicts whether a job posting will be **removed from the board** within the
model's horizon, from information available at the moment it is first seen.

*Removed is not filled.* A posting can be pulled, expire, or be reposted
elsewhere. The label is absence from the board, and no claim beyond that is
supported by it.

Every response carries the threshold the probability was compared against, the
horizon the prediction refers to, and whether board-level context was supplied —
a probability alone is not a decision.
"""


def artifact_path(override: Path | str | None = None) -> Path:
    """Explicit argument, then environment, then the default location."""
    if override is not None:
        return Path(override)
    return Path(os.environ.get(ARTIFACT_ENV, DEFAULT_ARTIFACT))


def create_app(artifact: Path | str | None = None) -> FastAPI:
    """Build the app.

    A factory rather than a module-level singleton so that a test can serve an
    artifact it froze in a temporary directory without reaching for environment
    variables — the same reason `freeze` takes a path.
    """
    path = artifact_path(artifact)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.predictor = Predictor.load(path)
            app.state.detail = None
        except (ArtifactError, OSError) as error:
            app.state.predictor = None
            app.state.detail = str(error)
        yield
        app.state.predictor = None

    app = FastAPI(
        title="shelf-life",
        description=DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.artifact_path = path

    @app.exception_handler(InvalidPayload)
    async def _invalid_payload(request: Request, error: InvalidPayload) -> JSONResponse:
        """The contract's refusals, as 422s.

        Pydantic has already checked the shape by the time a request reaches the
        contract, so anything raised there is about content — an unparseable
        salary string, a field that cannot be coerced. Still the caller's fault,
        still not a 500.
        """
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": str(error)},
        )

    def _predictor(app: FastAPI) -> Predictor:
        predictor = getattr(app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"no model loaded from {app.state.artifact_path}: "
                    f"{app.state.detail}. Build one with `python -m src.models.freeze`."
                ),
            )
        return predictor

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Is the process up, and does it have a model? Both, separately."""
        predictor = getattr(app.state, "predictor", None)
        if predictor is None:
            return HealthResponse(
                status="degraded",
                model_loaded=False,
                artifact=str(app.state.artifact_path),
                detail=getattr(app.state, "detail", None),
            )
        metadata = predictor.metadata
        return HealthResponse(
            status="ok",
            model_loaded=True,
            artifact=str(app.state.artifact_path),
            model=metadata.run_name,
            dataset=metadata.dataset,
            horizon_days=metadata.horizon_days,
            threshold=metadata.threshold,
            fitted_on=metadata.fitted_on,
            created_at=metadata.created_at,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(posting: PostingRequest) -> PredictionResponse:
        """Score one posting.

        The probability is the chance the posting leaves the board within the
        horizon named in the response — **not** the chance it is filled.
        """
        prediction = _predictor(app).predict(posting.payload(), t=posting.as_of)
        return PredictionResponse(**prediction.as_dict())

    @app.get("/contract")
    def contract() -> list[dict]:
        """Every field a caller may send, and whether they could know it.

        The audit as an endpoint. It is here because the interesting question
        about this model is not what it scores but what it is allowed to see,
        and that answer should be reachable without cloning the repository.
        """
        return describe().to_dict(orient="records")

    return app


app = create_app()
