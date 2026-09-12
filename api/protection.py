"""Two cheap limits for a public endpoint on a tenth of a CPU.

Authentication and rate limiting are non-goals of this build, and remain so
in the sense that neither is a *security* boundary here: there is no key, no
account, no per-user quota. What this module adds is the minimum that keeps
one careless or hostile caller from making the free instance useless for
everyone else, and it is honest about being that and nothing more.

**A request body limit, before parsing.** Pydantic already refuses a
2,000-posting body with a 422 — after JSON-decoding all of it. `MAX_BODY_BYTES`
is checked against `Content-Length` before the body is read, so an oversized
request costs the instance a header, not a parse. The limit is sized from the
largest legitimate request: a full page of 250 postings with every text field
at its cap is about 2 MB; twice that is the line.

**A per-IP budget on the expensive routes.** A 250-posting `/rank` is ~23 s of
CPU on the free instance; `/health` is nothing. So the expensive routes —
`/rank`, `/boards/*/score` — get a token bucket per client address:
`BURST` requests at once, refilling at `PER_MINUTE`. Over it is a 429 with
`Retry-After`, never a queue that slows every other caller. `/predict` and
`/health` are not limited: a single posting is cheap, and limiting `/health`
would break the UI's wake-up and the deploy verification.

**What it is not.** In-memory, per process, on one instance — a second
instance would have its own buckets, and a restart forgets them. The client
address is what the platform hands over (`X-Forwarded-For`'s first hop behind
Render's proxy), which a determined caller can vary. This is a courtesy limit
that protects the operator's own board ranking from an accidental loop or a
curious stranger's script; anything stronger is an API key, which is the
production step named in `docs/design.md` §7b-ii and deliberately not taken.
"""

from __future__ import annotations

import threading
import time

from starlette.requests import Request
from starlette.responses import JSONResponse

#: Twice the largest legitimate body: 250 postings with every text field at
#: `TEXT_MAX` is about 2 MB.
MAX_BODY_BYTES = 4 * 1024 * 1024

#: The expensive routes, and the budget each client address gets on them.
EXPENSIVE_PREFIXES = ("/rank", "/boards/")
EXPENSIVE_SUFFIXES = ("/rank", "/score")
BURST = 6
PER_MINUTE = 12.0


def is_expensive(path: str, method: str) -> bool:
    if method != "POST":
        return False
    if path == "/rank":
        return True
    return path.startswith("/boards/") and path.endswith(EXPENSIVE_SUFFIXES)


class TokenBuckets:
    """One bucket per client address; `BURST` tokens, refilled at `PER_MINUTE`."""

    def __init__(self, burst: int = BURST, per_minute: float = PER_MINUTE):
        self.burst = float(burst)
        self.rate = per_minute / 60.0
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def take(self, key: str, now: float | None = None) -> float:
        """Take one token. Returns 0 if allowed, else seconds until one refills."""
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._buckets.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                return 0.0
            self._buckets[key] = (tokens, now)
            return (1.0 - tokens) / self.rate

    def forget_older_than(self, seconds: float, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            stale = [k for k, (_, last) in self._buckets.items() if now - last > seconds]
            for key in stale:
                del self._buckets[key]


def client_address(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def install(app, buckets: TokenBuckets | None = None, max_body_bytes: int = MAX_BODY_BYTES):
    """Attach both limits to a FastAPI app as one HTTP middleware."""
    buckets = buckets or TokenBuckets()
    app.state.buckets = buckets

    @app.middleware("http")
    async def _protect(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > max_body_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": f"request body of {int(length):,} bytes exceeds the "
                    f"{max_body_bytes:,}-byte limit; send fewer postings per request"
                },
            )
        if is_expensive(request.url.path, request.method):
            wait = buckets.take(client_address(request))
            if wait > 0:
                return JSONResponse(
                    status_code=429,
                    headers={"Retry-After": str(max(1, int(wait + 0.999)))},
                    content={
                        "detail": f"too many expensive requests from this address; "
                        f"{BURST} at once, then {PER_MINUTE:.0f} a minute. Retry in "
                        f"{wait:.0f}s. There is no key to raise this: the service is "
                        "one free instance ranking one board at a time"
                    },
                )
        return await call_next(request)

    return app
