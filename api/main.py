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
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.runtime import process_age_seconds, rss_mb
from api.schemas import (
    BoardCreated,
    BoardCreateRequest,
    BoardPage,
    BoardProgress,
    BoardRankRequest,
    BoardRankResponse,
    HealthResponse,
    PostingRequest,
    PredictionResponse,
    RankedPosting,
    RankRequest,
    RankResponse,
)
from src.inference.artifact import DEFAULT_ARTIFACT, ArtifactError
from src.inference.boards import BoardExpired, BoardStore
from src.inference.contract import InvalidPayload, describe
from src.inference.predict import MAX_BATCH, Predictor

#: Where the artifact lives, overridable so a container can mount one elsewhere
#: and a test can point at one it built itself.
ARTIFACT_ENV = "SHELF_LIFE_ARTIFACT"

#: The release tag the image was built from. Set by the Dockerfile from
#: `MODEL_TAG`, absent everywhere else. Reported by `/health` so that a deploy
#: can be verified from outside as *the new one* rather than merely as *a live
#: one* — polling an endpoint that cannot tell you which version answered is how
#: a smoke test passes against the revision it was supposed to replace.
ARTIFACT_TAG_ENV = "SHELF_LIFE_ARTIFACT_TAG"

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


#: Written by the Dockerfile with the tag the build actually fetched. Absent
#: outside a container, which is why `/health` reports `null` locally rather
#: than guessing from `MODEL_TAG` — that file states an intention, and reporting
#: an intention as a fact is the drift this field exists to expose.
ARTIFACT_TAG_FILE = Path(__file__).resolve().parent.parent / "ARTIFACT_TAG"


def artifact_path(override: Path | str | None = None) -> Path:
    """Explicit argument, then environment, then the default location."""
    if override is not None:
        return Path(override)
    return Path(os.environ.get(ARTIFACT_ENV, DEFAULT_ARTIFACT))


def artifact_tag() -> str | None:
    """The release this image was built from, or `None` if it was not built from one."""
    from_env = os.environ.get(ARTIFACT_TAG_ENV)
    if from_env:
        return from_env
    try:
        return ARTIFACT_TAG_FILE.read_text().strip() or None
    except OSError:
        return None


def create_app(artifact: Path | str | None = None) -> FastAPI:
    """Build the app.

    A factory rather than a module-level singleton so that a test can serve an
    artifact it froze in a temporary directory without reaching for environment
    variables — the same reason `freeze` takes a path.
    """
    path = artifact_path(artifact)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Timed, because the unpickle is the one cost of a cold start that the
        # no-artifact baseline cannot see and the outside timing cannot isolate
        # (`docs/design.md` §7e). `/health` reports both numbers.
        started = perf_counter()
        try:
            app.state.predictor = Predictor.load(path)
            app.state.detail = None
        except (ArtifactError, OSError) as error:
            app.state.predictor = None
            app.state.detail = str(error)
        app.state.load_seconds = perf_counter() - started
        app.state.ready_after_seconds = process_age_seconds()
        yield
        app.state.predictor = None

    app = FastAPI(
        title="shelf-life",
        description=DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.artifact_path = path
    # The live boards. In memory, on this instance, with a TTL — the honest
    # shape on a free tier that sleeps, and said so in every response.
    app.state.boards = BoardStore()

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

    @app.exception_handler(BoardExpired)
    async def _board_expired(request: Request, error: BoardExpired) -> JSONResponse:
        """A board that is gone is a 404 that says to start over, not a 500."""
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(error)})

    def _predictor(app: FastAPI) -> Predictor:
        predictor = getattr(app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                # The loader's own message already names the artifact and the
                # command that builds one. Appending a second copy of the
                # instruction is how a 503 body ends up saying the same sentence
                # twice to the person who most needs to read it once.
                detail=f"no model loaded: {app.state.detail}",
            )
        return predictor

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Is the process up, and does it have a model? Both, separately."""
        predictor = getattr(app.state, "predictor", None)
        cost = {
            "load_seconds": getattr(app.state, "load_seconds", None),
            "ready_after_seconds": getattr(app.state, "ready_after_seconds", None),
            "rss_mb": rss_mb(),
        }
        if predictor is None:
            return HealthResponse(
                status="degraded",
                model_loaded=False,
                artifact=str(app.state.artifact_path),
                artifact_tag=artifact_tag(),
                detail=getattr(app.state, "detail", None),
                **cost,
            )
        metadata = predictor.metadata
        return HealthResponse(
            status="ok",
            model_loaded=True,
            artifact=str(app.state.artifact_path),
            artifact_tag=artifact_tag(),
            **cost,
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

        The probability is the chance the posting is removed from the board
        within the horizon named in the response — **not** the chance it is
        filled. The response says so itself, in `predicts`.
        """
        prediction = _predictor(app).predict(posting.payload(), t=posting.as_of)
        return PredictionResponse(**prediction.as_dict())

    @app.post("/rank", response_model=RankResponse)
    def rank(request: RankRequest) -> RankResponse:
        """Score many postings and order them, applying the alert budget.

        The shape the operating point was designed for. The frozen threshold is
        the score at which exactly `budget` postings are flagged — a rank
        statistic over a day's board — so `/predict` applies it to one posting in
        isolation while this applies it to a list, which is where it came from.

        Postings come back in the order they were sent; `rank` carries the
        ordering. `threshold_source` says whether `watch` was decided by the
        model's frozen operating point or by this batch's budget-th score, and
        the two are never silently interchanged.

        **This is where board context lives.** `/predict` takes only what a
        person holding one advert can know. Here, with `is_board_snapshot`,
        the caller declares the batch is one whole board and the service
        derives `board_size_at_t`, `n_same_title_on_board`, `n_same_req_on_board`
        and (given `previous_board_size`) `board_growth` from it by the panel's
        own definitions. Undeclared, a batch is not the board — fifty postings
        must not manufacture a board of fifty — and the four are imputed, or
        used if a posting carries them.
        """
        batch = _predictor(app).rank(
            [posting.payload() for posting in request.postings],
            t=request.as_of,
            budget=request.budget,
            board_snapshot=request.is_board_snapshot,
            previous_board_size=request.previous_board_size,
        )
        return RankResponse(
            postings=[RankedPosting(**item.as_dict()) for item in batch.postings],
            threshold_applied=batch.threshold_applied,
            threshold_source=batch.threshold_source,
            budget=batch.budget,
            horizon_days=batch.horizon_days,
            model=batch.model,
            dataset=batch.dataset,
            t=batch.t,
            board_context_source=batch.board_context_source,
            board_size=batch.board_size,
        )

    # --- a board as one logical collection ------------------------------------

    @app.post("/boards", response_model=BoardCreated, status_code=status.HTTP_201_CREATED)
    def create_board(request: BoardCreateRequest) -> BoardCreated:
        """Open a board to upload in pages, score in pages, and rank once.

        The whole-board ranking cannot be one request here — 1,150 postings on
        a tenth of a CPU is about 106 s against a 90 s rule — so it is several,
        and this is the collection they share. The prediction instant and the
        snapshot declaration are fixed now; `/boards/{id}/postings` appends up
        to `page_size` at a time; `/boards/{id}/score` scores the next page,
        deriving board context over the whole board first when it is a
        snapshot; `/boards/{id}/rank` orders the whole board by the same rule
        `/rank` applies to a batch. The client pages and loops; it never
        rebuilds the ranking, and never sees a probability until the ranking.

        Boards live in memory on one instance for `ttl_seconds`, and are gone
        when the service sleeps or redeploys — a 404 that says to start over.
        """
        _predictor(app)
        board = app.state.boards.create(
            Predictor.moment(request.as_of),
            is_snapshot=request.is_board_snapshot,
            previous_board_size=request.previous_board_size,
        )
        return BoardCreated(
            board_id=board.board_id, ttl_seconds=app.state.boards._ttl, page_size=MAX_BATCH
        )

    @app.post("/boards/{board_id}/postings", response_model=BoardProgress)
    def add_postings(board_id: str, page: BoardPage) -> BoardProgress:
        """Append a page of postings. Refused once scoring has begun."""
        board = app.state.boards.get(board_id)
        board.add([posting.payload() for posting in page.postings])
        return BoardProgress(
            board_id=board_id, total=board.total, scored=board.scored, done=board.done
        )

    @app.post("/boards/{board_id}/score", response_model=BoardProgress)
    def score_board(board_id: str) -> BoardProgress:
        """Score the next unscored page; call until `done`."""
        board = app.state.boards.get(board_id)
        board.score_next(_predictor(app))
        return BoardProgress(
            board_id=board_id, total=board.total, scored=board.scored, done=board.done
        )

    @app.post("/boards/{board_id}/rank", response_model=BoardRankResponse)
    def rank_board(board_id: str, request: BoardRankRequest) -> BoardRankResponse:
        """The ranking over the whole board. Refused until every page is scored."""
        board = app.state.boards.get(board_id)
        batch = board.rank(_predictor(app), request.budget)
        return BoardRankResponse(
            postings=[RankedPosting(**item.as_dict()) for item in batch.postings],
            threshold_applied=batch.threshold_applied,
            threshold_source=batch.threshold_source,
            budget=batch.budget,
            horizon_days=batch.horizon_days,
            model=batch.model,
            dataset=batch.dataset,
            t=batch.t,
            board_context_source=batch.board_context_source,
            board_size=batch.board_size,
            board_id=board_id,
            pages_scored=-(-board.total // MAX_BATCH),
        )

    @app.delete("/boards/{board_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_board(board_id: str) -> None:
        """Forget a board early. Idempotent."""
        app.state.boards.delete(board_id)

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
