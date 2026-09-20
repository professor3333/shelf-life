"""The process's own account of what it cost to start — `api/runtime.py`.

`/health` reports these beside the outside timing so a slow cold start arrives
with its decomposition. The `/proc` branch is the one that runs in the image and
the one the test runner (macOS) never reaches, so it is exercised here with
hand-written `/proc` files: the arithmetic — field 22 of `/proc/self/stat`,
found after the last `)` because the command name may itself contain one — is
where a wrong index would report a plausible number that is nonetheless not the
process's age.
"""

from __future__ import annotations

import builtins
import io
import sys

import pytest

from api import runtime

# A real-looking stat line whose command name contains both a space and a `)`,
# which is exactly why the parse splits on the *last* `)` and not the first.
# After the `)`: state is field 3 (index 0), so field 22, starttime, is index 19.
_AFTER_PAREN = ["S", "1", "4242", "4242", "0", "-1", "4194560", *(["7"] * 12), "123456"]
STAT = "4242 (uvicorn (main)) " + " ".join([*_AFTER_PAREN, *(["0"] * 30)])
assert _AFTER_PAREN.index("123456") == 19


def _fake_proc(monkeypatch, stat: str, uptime: str, ticks: int = 100) -> None:
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/self/stat":
            return io.StringIO(stat)
        if path == "/proc/uptime":
            return io.StringIO(uptime)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(runtime.os, "sysconf", lambda name: ticks)


def test_process_age_reads_starttime_from_after_the_last_paren(monkeypatch):
    """starttime is 123456 ticks at 100 Hz = 1234.56 s after boot; uptime 2000 s."""
    _fake_proc(monkeypatch, STAT, "2000.00 8000.00\n")
    assert runtime.process_age_seconds() == pytest.approx(2000.0 - 1234.56)


def test_a_malformed_stat_falls_back_to_the_module_age(monkeypatch):
    """Too few fields after the `)` is an IndexError, and the fallback is the
    import-time clock — not an exception from `/health`."""
    _fake_proc(monkeypatch, "1 (short) S 1 2", "2000.00 8000.00\n")
    age = runtime.process_age_seconds()
    assert 0.0 <= age < 3600.0


def test_an_unreadable_uptime_falls_back_too(monkeypatch):
    _fake_proc(monkeypatch, STAT, "not-a-number\n")
    age = runtime.process_age_seconds()
    assert 0.0 <= age < 3600.0


def test_no_proc_at_all_falls_back(monkeypatch):
    def no_proc(path, *args, **kwargs):
        raise FileNotFoundError(path)

    monkeypatch.setattr(builtins, "open", no_proc)
    assert runtime.process_age_seconds() >= 0.0


def test_peak_rss_is_reported_in_mib_on_both_platforms(monkeypatch):
    """`ru_maxrss` is kilobytes on Linux and bytes on macOS; the function is the
    place that disagreement is absorbed, so it is checked on both readings."""

    class Usage:
        ru_maxrss = 300 * 1024 * 1024  # 300 MiB, as macOS reports it (bytes)

    monkeypatch.setattr(runtime.resource, "getrusage", lambda who: Usage())
    monkeypatch.setattr(sys, "platform", "darwin")
    assert runtime.rss_mb() == pytest.approx(300.0)

    Usage.ru_maxrss = 300 * 1024  # the same 300 MiB, as Linux reports it (kB)
    monkeypatch.setattr(sys, "platform", "linux")
    assert runtime.rss_mb() == pytest.approx(300.0)
