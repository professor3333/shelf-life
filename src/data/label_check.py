"""Does "removed from the board" mean what the label says it means?

`src/data/label_audit.py` measures the label against *itself* — relisting rates,
board stability, closure dispersion, the concentration of closures on postings
the crawl barely held. All of it is internal evidence, and its own closing
section says what it cannot do: establish whether a posting the label calls
closed is actually gone. That needs the board.

This is that check. It takes postings the panel labels **removed**, asks the
source whether they are still listed, and counts the answers. It also asks the
same question of a **control** group — postings that were on the board at the
final crawl — because agreement on the positives alone proves nothing: an
instrument that reports "gone" for every URL would agree perfectly with a label
that was pure noise. The control is what makes the positives readable.

**Two instruments, one per source family, and only one of them was obvious.**

* **Greenhouse** exposes a documented public board endpoint,
  `boards-api.greenhouse.io/v1/boards/<slug>/jobs`, which returns the whole live
  board. Membership of a job id in that list is the check, and it costs *one*
  request per board rather than one per posting.
* **python.org** answers per posting: `200` for a live job, `404` for one that is
  gone.

The obvious instrument for Greenhouse — fetching the posting's own URL — is
**wrong, and quietly so**. `job-boards.greenhouse.io/<slug>/jobs/<id>` answers
`200` for both a live job and a dead one, redirecting to the board index with
`?error=true` in the second case *and in cases where the job is plainly still
listed*. A check built on it reports "removed" for everything, agrees with the
label perfectly, and proves nothing. It was tried first and the board API is
what replaced it.

**Time is the caveat and it runs one way.** The label describes a window that
closed before the last crawl; this asks about *now*. A posting genuinely removed
then is still absent now, so a positive that reads "gone" is consistent. But a
posting still listed *now* cannot have been removed then — ids are not reused —
so **"still listed" is a label error**, not a timing artifact. The control
measures the drift in the other direction: postings that were up at the final
crawl and have since gone are the rate at which "absent now" happens anyway, and
that rate is what the positives' agreement has to beat.

**Removed is still not filled.** Nothing here can see why a posting left. A
verified removal is a verified *removal*; the README's caveat stands unchanged.

Politeness: `robots.txt` allows both hosts (Greenhouse disallows nothing;
python.org excludes only `/~guido/orlijn/` and `/webstats/`). Requests identify
the project, are spaced by `DELAY_SECONDS`, and the Greenhouse arm costs one
request per board however many postings are checked.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.features.assemble import horizon_banner, panel_path
from src.models import provenance
from src.models.train_baseline import _table

DEFAULT_PANEL = panel_path()
DEFAULT_REPORT = Path("reports/label_check.md")

#: Identifies the project and offers a way to complain. A check that anonymises
#: itself while asking someone else's server about their data has the ethics the
#: wrong way round.
USER_AGENT = "shelf-life-label-audit/0.1 (+https://github.com/professor3333/shelf-life)"

#: Seconds between requests. The Greenhouse arm makes one request per board, so
#: this mostly paces the python.org arm; generous rather than tuned, because the
#: whole check is a few dozen requests and there is nothing to gain by hurrying.
DELAY_SECONDS = 2.0

GREENHOUSE_BOARD_API = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"


class Unverifiable(RuntimeError):
    """The source could not be asked, so the posting is counted as unchecked."""


@dataclass(frozen=True)
class Verdict:
    """One posting, what the label said, and what the board says now."""

    source: str
    source_id: str
    url: str
    label: str
    listed_now: bool | None
    verdict: str
    note: str = ""


def _get(url: str, timeout: float = 45.0) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, b""
    except (urllib.error.URLError, TimeoutError) as error:  # pragma: no cover - network
        raise Unverifiable(str(error)) from error


def greenhouse_board(slug: str, get: Callable[[str], tuple[int, bytes]] = _get) -> dict[str, str]:
    """Every job currently listed on one Greenhouse board: id -> title.

    One request for the whole board. The title comes back too, which is what
    makes relisting detectable: a posting gone under its own id whose title
    reappears under another is a re-post, not a removal that stuck.
    """
    status, body = get(GREENHOUSE_BOARD_API.format(slug=slug))
    if status != 200:
        raise Unverifiable(f"board API returned {status} for {slug}")
    try:
        payload = json.loads(body)
    except ValueError as error:
        raise Unverifiable(f"board API returned unparseable JSON for {slug}") from error
    return {str(job["id"]): str(job.get("title", "")) for job in payload.get("jobs", [])}


def python_org_listed(url: str, get: Callable[[str], tuple[int, bytes]] = _get) -> bool:
    """Is this python.org posting still up? 200 yes, 404 no."""
    status, _ = get(url)
    if status == 200:
        return True
    if status == 404:
        return False
    raise Unverifiable(f"unexpected status {status} for {url}")


def sample_for_check(
    panel: pd.DataFrame,
    alive_at_final_crawl: set[str],
    per_arm: int = 60,
    seed: int = 0,
) -> pd.DataFrame:
    """Postings the label calls removed, and a control that was up at the last crawl.

    **Both arms, always.** Agreement on the positives alone proves nothing: an
    instrument that answered "gone" for every URL would agree perfectly with a
    label that was pure noise. The control is what the positives' agreement has
    to beat, and it is drawn from postings the final crawl saw — so "absent now"
    in that arm is the rate at which a live posting disappears anyway in the days
    since, which is the drift the positives have to clear.

    Sampled per source in proportion to that source's share of the arm. The
    sources differ in exactly the way this check is about — different parsers
    against different page structures — so one row each would under-weight the
    board carrying three quarters of the closures.
    """
    labelled = panel[panel["label_observable"]].copy()
    labelled["key"] = labelled["source"].astype(str) + "|" + labelled["source_id"].astype(str)

    removed = labelled[labelled["y"] == 1].drop_duplicates("key")
    frames = [_stratified(removed, per_arm, seed).assign(arm="removed")]

    survivors = labelled[labelled["y"] == 0].drop_duplicates("key")
    survivors = survivors[survivors["key"].isin(alive_at_final_crawl)]
    if len(survivors):
        frames.append(
            _stratified(survivors, per_arm, seed).assign(arm="still listed at last crawl")
        )
    return pd.concat(frames, ignore_index=True)


def _stratified(rows: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
    if len(rows) <= total:
        return rows.copy()
    shares = rows["source"].value_counts(normalize=True)
    picked = []
    for source, share in shares.items():
        want = max(1, round(total * share))
        block = rows[rows["source"] == source]
        picked.append(block.sample(min(want, len(block)), random_state=seed))
    return pd.concat(picked, ignore_index=True).head(total)


def verify(
    sample: pd.DataFrame,
    get: Callable[[str], tuple[int, bytes]] = _get,
    delay: float = DELAY_SECONDS,
) -> pd.DataFrame:
    """Ask each source about each posting, and record what it said.

    Greenhouse boards are fetched once each and reused across every posting from
    that board, which is why a check over a hundred postings costs single-digit
    requests to Greenhouse.
    """
    boards: dict[str, dict[str, str] | None] = {}
    verdicts: list[Verdict] = []

    for row in sample.itertuples():
        source, source_id, url = str(row.source), str(row.source_id), str(row.url)
        label = str(row.arm)
        try:
            if source.startswith("greenhouse:"):
                slug = source.split(":", 1)[1]
                if slug not in boards:
                    boards[slug] = greenhouse_board(slug, get)
                    time.sleep(delay)
                listing = boards[slug]
                if listing is None:  # pragma: no cover - defensive
                    raise Unverifiable(f"no listing for {slug}")
                listed = source_id in listing
                note = ""
                if not listed:
                    title = str(getattr(row, "title", "") or "")
                    matches = [i for i, t in listing.items() if t and t == title]
                    if matches:
                        note = f"title still listed under id {matches[0]}"
                verdicts.append(_verdict(source, source_id, url, label, listed, note))
            elif source == "python_org":
                listed = python_org_listed(url, get)
                time.sleep(delay)
                verdicts.append(_verdict(source, source_id, url, label, listed))
            else:
                verdicts.append(
                    Verdict(source, source_id, url, label, None, "unverifiable", "no instrument")
                )
        except Unverifiable as error:
            verdicts.append(
                Verdict(source, source_id, url, label, None, "unverifiable", str(error))
            )

    return pd.DataFrame([vars(v) for v in verdicts])


def _verdict(source, source_id, url, label, listed: bool, note: str = "") -> Verdict:
    if label == "removed":
        if listed:
            verdict = "still listed — label wrong"
        elif note:
            verdict = "removed, title relisted"
        else:
            verdict = "genuine removal"
    else:
        verdict = "still listed" if listed else "gone since the last crawl"
    return Verdict(source, source_id, url, label, listed, verdict, note)


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% interval on a proportion, by Wilson's method.

    Not the normal approximation. At 59 of 60 the normal interval runs past 1,
    which is not a bound on a proportion — and this check's whole point is that
    the number near the edge is the one being reported.
    """
    if trials <= 0:
        return (float("nan"), float("nan"))
    p_hat = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p_hat + z**2 / (2 * trials)) / denominator
    spread = z * ((p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) ** 0.5) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Counts per arm and verdict — the table the conclusion is read off."""
    if results.empty:
        return pd.DataFrame()
    table = results.groupby(["label", "verdict"]).size().rename("postings").reset_index()
    totals = results.groupby("label").size().rename("arm_total")
    table = table.join(totals, on="label")
    table["share"] = (table["postings"] / table["arm_total"]).round(4)
    return table


def write_report(path: Path, results: pd.DataFrame, prov, horizon: str, sampled: int) -> None:
    """Write `reports/label_check.md`."""
    summary = summarise(results)
    removed = results[results["label"] == "removed"]
    control = results[results["label"] != "removed"]

    lines = [
        "# Label check — verified against the board",
        "",
        *provenance.header(prov, "python -m src.data.label_check", horizon),
        "**What this answers.** `label_validity.md` measures the label against itself.",
        "Its closing section says what it cannot do: establish whether a posting the label",
        "calls closed is actually gone. This asks the board.",
        "",
        "**Removed is still not filled.** Nothing here can see *why* a posting left. A",
        "verified removal is a verified removal, and the README's caveat stands unchanged.",
        "",
        "## What was asked, and of whom",
        "",
        f"{sampled} postings, in two arms. The control arm is not decoration: an instrument",
        "that answered *gone* for every URL would agree perfectly with a label that was pure",
        "noise, so the positives' agreement is only worth what it beats.",
        "",
        "| source family | instrument |",
        "|---|---|",
        "| Greenhouse | `boards-api.greenhouse.io/v1/boards/<slug>/jobs` — one request per",
        "board, membership of the job id in the live listing |",
        "| python.org | the posting's own URL: `200` listed, `404` gone |",
        "",
        "**The obvious Greenhouse instrument is wrong and quietly so.** Fetching",
        "`job-boards.greenhouse.io/<slug>/jobs/<id>` returns `200` for live *and* dead",
        "postings, redirecting to the board index with `?error=true` in cases where the job",
        "is plainly still listed. A check built on it reports *removed* for everything and",
        "agrees with the label perfectly. It was tried first; the board API replaced it.",
        "",
        "## Results",
        "",
        _table(summary, list(summary.columns)) if not summary.empty else "_Nothing checked._",
        "",
    ]

    if not removed.empty:
        genuine = int((removed["verdict"] == "genuine removal").sum())
        wrong = int((removed["verdict"] == "still listed — label wrong").sum())
        relisted = int((removed["verdict"] == "removed, title relisted").sum())
        unverifiable = int((removed["verdict"] == "unverifiable").sum())
        checked = len(removed) - unverifiable
        # A relisted title is still a removal of *that posting*: it is gone under
        # its own id. Counting it against the label would be scoring the label on
        # a question it does not ask.
        gone = genuine + relisted
        rate = gone / checked if checked else float("nan")
        low, high = wilson(gone, checked)
        lines += [
            "### The positives",
            "",
            f"Of {checked} postings the label calls removed, **{gone} are gone from the board "
            f"under their own id** — {rate:.1%} (95% Wilson {low:.1%}–{high:.1%}).",
            "",
            f"{relisted} of those {gone} have the same title listed under a *different* id. "
            "The posting was removed; the role is still being advertised. That is a removal by "
            "the label's definition and a reminder of what the definition is worth: **removed "
            "is not filled**, and a relisted title is the clearest case of the two coming "
            "apart.",
            "",
            f"{wrong} is still listed under its own id. That one the label cannot survive being "
            "right about — ids are not reused, so a posting listed now was not removed then. It "
            "is a missed observation: the crawl lost sight of a posting that never left.",
            "",
        ]
        if unverifiable:
            lines += [f"{unverifiable} could not be checked and are excluded from the rate.", ""]

    if not control.empty:
        still = int((control["verdict"] == "still listed").sum())
        gone = int((control["verdict"] == "gone since the last crawl").sum())
        checked = still + gone
        drift = gone / checked if checked else float("nan")
        lines += [
            "### The control",
            "",
            f"Of {checked} postings that were on the board at the final crawl, {still} are "
            f"still listed and {gone} have gone since — a drift of {drift:.1%}.",
            "",
            "That is the number the positives have to beat, and the reason the control arm "
            "exists. *Absent now* happens to live postings too, just by time passing; if it "
            "happened at anything like the rate the positives show, their agreement would be "
            "the calendar rather than the label.",
            "",
        ]

    lines += [
        "## What this cannot show",
        "",
        "The label describes a window that closed before the last crawl; this asks about",
        "*now*, and the two are days apart. The asymmetry runs one way and is why the check",
        "is worth taking: a posting genuinely removed then is still absent now, so agreement",
        "is consistent rather than proof — but a posting **still listed now cannot have been",
        "removed then**, so a disagreement is a label error outright.",
        "",
        "It also cannot see a posting that was removed and relisted under the same id, or",
        "one pulled and reinstated between crawls. `label_validity.md` bounds the second",
        "with the resurrection window.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:  # pragma: no cover - thin CLI over the network
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--per-arm", type=int, default=60)
    parser.add_argument("--delay", type=float, default=DELAY_SECONDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/label_check_results.csv"),
        help="where the raw verdicts are written. Gitignored: it is evidence for one "
        "report, not a source artifact",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="rewrite the report from the last run's verdicts without asking the boards "
        "again. The right default for editing the prose: re-fetching to change a "
        "sentence spends somebody else's bandwidth on nothing",
    )
    args = parser.parse_args()

    from src.data.load import load_snapshot
    from src.data.snapshot import latest_snapshot

    panel = pd.read_parquet(args.panel)
    frames = load_snapshot(latest_snapshot())
    jobs, observations, runs = frames["jobs"], frames["job_observations"], frames["runs"]
    final = runs.sort_values("started_at").groupby("source").tail(1)
    seen = observations[observations["run_id"].isin(set(final["id"]))]
    alive = jobs[jobs["id"].isin(set(seen["job_id"]))]
    alive_keys = set(alive["source"].astype(str) + "|" + alive["source_id"].astype(str))

    if args.from_cache and args.cache.exists():
        results = pd.read_csv(args.cache)
        sample_size = len(results)
        print(f"reusing {sample_size} verdicts from {args.cache} — no requests made")
    else:
        sample = sample_for_check(panel, alive_keys, args.per_arm, args.seed)
        sample_size = len(sample)
        print(f"checking {sample_size} postings ({sample['arm'].value_counts().to_dict()})")
        results = verify(sample, delay=args.delay)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.cache, index=False)
    print(results["verdict"].value_counts().to_string())

    write_report(
        args.out,
        results,
        provenance.collect(args.panel, len(panel), provenance.REAL),
        horizon_banner(panel),
        sample_size,
    )
    print(f"wrote -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
