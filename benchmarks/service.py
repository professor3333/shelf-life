"""How the service behaves under load: concurrency, latency shape, memory, abuse.

    python -m benchmarks.service http://localhost:8000              # a local container
    python -m benchmarks.service http://localhost:8000 --label local
    python -m benchmarks.service "$SHELF_LIFE_API" --label render --concurrency 4

`MAX_BATCH` came from one measurement — 250 postings, one request, one core.
This asks the questions that measurement did not: what happens when several
callers arrive at once, what the latency *distribution* looks like once the
process is warm, whether memory moves while ranking maximum-size payloads,
and what a malformed or oversized request does to the process. Each section
is a behaviour with a pass condition, and the numbers are printed beside it
so a reader can see what "passed" meant on this machine.

**Absolute figures are not Render's.** A laptop core is roughly ten times the
free instance's tenth of a CPU, and a benchmark against a sleeping free
instance measures its wake, not the service. Run it against the public URL
only when the service is warm, with low concurrency, and read the shape:
no 5xx, no memory growth, a p95 that is the p50 plus a queue rather than a
cliff. The report says which target it measured.

**There is no rate limit, by design**, and the last section measures what
that means rather than pretending otherwise: a burst of requests queues at
the single worker and every one is answered, slowly. `docs/design.md` §7b;
the project's non-goals list authentication and rate limiting explicitly.

Written as functions over a `post(path, body)` callable so the suite can run
the same code in-process against the app with no socket.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

Post = Callable[[str, dict], tuple[int, float, dict | None]]
"""(status code, seconds, parsed body or None) for a POST of `body` to `path`."""

POSTING = {
    "title": "Senior Data Engineer",
    "location": "Berlin",
    "salary_raw": "120000 - 160000 USD",
    "departments": "Eng",
    "offices": "HQ",
    "content_chars": 1400,
    "first_published": "2026-08-20T00:00:00Z",
}
AS_OF = "2026-09-14T03:45:00Z"


def _board(n: int) -> list[dict]:
    return [
        {**POSTING, "title": f"{POSTING['title']} {i}", "content_chars": 400 + (i * 37) % 2600}
        for i in range(n)
    ]


@dataclass
class Section:
    name: str
    passed: bool
    detail: str
    numbers: dict[str, float] = field(default_factory=dict)


def _percentiles(seconds: list[float]) -> dict[str, float]:
    if not seconds:
        return {}
    ordered = sorted(seconds)
    q = statistics.quantiles(ordered, n=100) if len(ordered) >= 2 else ordered * 99

    return {
        "n": float(len(ordered)),
        "p50_ms": 1000 * (statistics.median(ordered)),
        "p95_ms": 1000 * (q[94] if len(ordered) >= 2 else ordered[0]),
        "max_ms": 1000 * ordered[-1],
    }


def warm_predict(post: Post, requests: int = 50) -> Section:
    """Repeated single predictions after warm-up: the latency shape."""
    body = {**POSTING, "as_of": AS_OF}
    for _ in range(3):
        post("/predict", body)
    times, codes = [], []
    for _ in range(requests):
        code, seconds, _ = post("/predict", body)
        times.append(seconds)
        codes.append(code)
    numbers = _percentiles(times)
    ok = all(c == 200 for c in codes)
    return Section(
        "warm /predict, sequential",
        ok and numbers["p95_ms"] < 20 * numbers["p50_ms"],
        f"{requests} requests, all 200" if ok else f"non-200 among {codes}",
        numbers,
    )


def concurrent_predict(post: Post, concurrency: int = 8, requests: int = 64) -> Section:
    """Several callers at once. Pass: every request answered 200, no 5xx."""
    body = {**POSTING, "as_of": AS_OF}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda _: post("/predict", body), range(requests)))
    codes = [r[0] for r in results]
    numbers = _percentiles([r[1] for r in results])
    numbers["concurrency"] = float(concurrency)
    return Section(
        f"concurrent /predict ×{concurrency}",
        all(c == 200 for c in codes),
        f"{requests} requests, {codes.count(200)} × 200, {sum(c >= 500 for c in codes)} × 5xx",
        numbers,
    )


def concurrent_rank(post: Post, batch: int, concurrency: int = 4, requests: int = 8) -> Section:
    """Maximum-size batches, several at once: the heaviest thing a caller can send."""
    body = {"postings": _board(batch), "budget": 20, "as_of": AS_OF}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda _: post("/rank", body), range(requests)))
    codes = [r[0] for r in results]
    numbers = _percentiles([r[1] for r in results])
    numbers["batch"] = float(batch)
    numbers["concurrency"] = float(concurrency)
    ok = all(c == 200 for c in codes) and all(
        r[2] is not None and sum(p["watch"] for p in r[2]["postings"]) == 20 for r in results
    )
    return Section(
        f"concurrent /rank ×{concurrency}, {batch} postings each",
        ok,
        f"{requests} requests, {codes.count(200)} × 200; every ranking flagged exactly 20"
        if ok
        else f"codes {codes}",
        numbers,
    )


def memory_while_ranking(
    get_health: Callable[[], dict], post: Post, batch: int, rounds: int = 5
) -> Section:
    """Peak RSS before and after repeated maximum-size rankings. Pass: flat."""
    before = float(get_health().get("rss_mb") or float("nan"))
    body = {"postings": _board(batch), "budget": 20, "as_of": AS_OF}
    for _ in range(rounds):
        post("/rank", body)
    after = float(get_health().get("rss_mb") or float("nan"))
    growth = after - before
    return Section(
        f"peak RSS across {rounds} rankings of {batch}",
        growth == growth and growth < 50.0,
        f"{before:.0f} MB → {after:.0f} MB ({growth:+.0f} MB); a leak would climb every round",
        {"rss_before_mb": before, "rss_after_mb": after, "growth_mb": growth},
    )


def malformed_and_oversized(post: Post, batch: int) -> Section:
    """Abuse: one over the cap, a huge string, a huge body, nonsense. Pass: 4xx, never 5xx."""
    cases = {
        "one over the batch cap": {"postings": _board(batch + 1), "as_of": AS_OF},
        "a 5 MB title": {"title": "x" * 5_000_000},
        "a 2,000-posting body": {"postings": _board(2000)},
        "nonsense fields": {"title": "x", "salary": "lots", "when": "soon"},
        "wrong types": {"title": 42, "content_chars": "many"},
    }
    outcomes = {}
    for name, body in cases.items():
        path = "/rank" if "postings" in body else "/predict"
        code, seconds, _ = post(path, body)
        outcomes[name] = (code, seconds)
    ok = all(400 <= code < 500 for code, _ in outcomes.values())
    detail = "; ".join(
        f"{name}: {code} in {seconds * 1000:.0f} ms" for name, (code, seconds) in outcomes.items()
    )
    return Section("malformed and oversized requests", ok, detail, {})


def burst_without_a_rate_limit(post: Post, burst: int = 32) -> Section:
    """What no rate limit means: a burst queues at the worker and is answered."""
    body = {**POSTING, "as_of": AS_OF}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=burst) as pool:
        results = list(pool.map(lambda _: post("/predict", body), range(burst)))
    wall = time.perf_counter() - started
    codes = [r[0] for r in results]
    numbers = _percentiles([r[1] for r in results])
    numbers["burst"] = float(burst)
    numbers["wall_s"] = wall
    return Section(
        f"a burst of {burst} with no rate limit",
        all(c == 200 for c in codes),
        f"all {burst} answered in {wall:.2f} s wall; none refused, none 5xx — there is no "
        "limit to hit, so a burst costs everyone latency rather than anyone an error",
        numbers,
    )


def run(
    post: Post, get_health: Callable[[], dict], batch: int, concurrency: int, light: bool = False
) -> list[Section]:
    n = 8 if light else 50
    return [
        warm_predict(post, requests=n),
        concurrent_predict(post, concurrency=concurrency, requests=2 * n),
        concurrent_rank(post, batch, concurrency=min(concurrency, 4), requests=4 if light else 8),
        memory_while_ranking(get_health, post, batch, rounds=2 if light else 5),
        malformed_and_oversized(post, batch),
        burst_without_a_rate_limit(post, burst=8 if light else 32),
    ]


def render(sections: list[Section], target: str, label: str, health: dict) -> str:
    lines = [
        "# Service benchmark — behaviour under load",
        "",
        "| | |",
        "|---|---|",
        "| Generated by | `python -m benchmarks.service` |",
        f"| Data | target `{target}` (`{label}`) · model `{health.get('model')}` on the "
        f"`{health.get('dataset')}` panel · release `{health.get('artifact_tag') or '(none)'}` |",
        f"| Code | measured on {datetime.now(UTC).date().isoformat()}; absolute figures are "
        f"this target's, not the free instance's unless the label says so |",
        "",
        "_Regenerate rather than edit._",
        "",
        "Each section is a behaviour with a pass condition; the numbers beside it say what",
        "passing meant here. Absolute latencies on a laptop core are roughly ten times",
        "faster than the free instance's tenth of a CPU — read the shape, not the number.",
        "",
        "| section | pass | detail |",
        "|---|---|---|",
    ]
    for s in sections:
        lines.append(f"| {s.name} | {'yes' if s.passed else '**NO**'} | {s.detail} |")
    lines += [
        "",
        "## Numbers",
        "",
        "| section | " + " | ".join(sorted({k for s in sections for k in s.numbers})) + " |",
    ]
    keys = sorted({k for s in sections for k in s.numbers})
    lines.append("|---|" + "---|" * len(keys))
    for s in sections:
        lines.append(
            f"| {s.name} | "
            + " | ".join(f"{s.numbers[k]:.1f}" if k in s.numbers else "—" for k in keys)
            + " |"
        )
    lines += [
        "",
        "## What is deliberately not here",
        "",
        "A rate limit. The service has none (`docs/design.md` §7b; the non-goals), so the",
        "last section measures what that costs — a burst queues at the single worker and",
        "every request is answered, slowly — rather than testing a limit that does not",
        "exist. A public free instance with no limit can be made slow by one caller; it",
        "cannot be made to answer wrongly, which is what the sections above check.",
        "",
    ]
    return "\n".join(lines)


def _http(base_url: str) -> tuple[Post, Callable[[], dict]]:
    import requests

    base = base_url.rstrip("/")

    def post(path: str, body: dict) -> tuple[int, float, dict | None]:
        started = time.perf_counter()
        try:
            response = requests.post(f"{base}{path}", json=body, timeout=180)
        except requests.RequestException:
            return 599, time.perf_counter() - started, None
        seconds = time.perf_counter() - started
        try:
            parsed = response.json()
        except ValueError:
            parsed = None
        return response.status_code, seconds, parsed

    def get_health() -> dict:
        return requests.get(f"{base}/health", timeout=60).json()

    return post, get_health


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--label", default="local")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--light", action="store_true", help="a quick pass with few requests")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    post, get_health = _http(args.url)
    health = get_health()
    if not health.get("model_loaded"):
        print("no model loaded at the target; nothing to benchmark", file=sys.stderr)
        return 2
    batch = int(health.get("rank_max_batch") or 250)
    sections = run(post, get_health, batch, args.concurrency, light=args.light)
    out = args.out or Path("reports") / f"benchmark_{args.label}.md"
    out.write_text(render(sections, args.url, args.label, health))
    for s in sections:
        print(f"{'ok  ' if s.passed else 'FAIL'} {s.name}: {s.detail}")
    print(f"wrote -> {out}")
    return 0 if all(s.passed for s in sections) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
