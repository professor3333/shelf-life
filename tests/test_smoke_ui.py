"""A rejected wake request must fail immediately, without a network or a wait."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("transport_failure", [False, True])
def test_rejected_wake_request_reports_the_cause_without_polling(tmp_path, transport_failure):
    curl = tmp_path / "curl"
    curl.write_text(
        f"#!{sys.executable}\n"
        "import sys\nfrom pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "if args[-1].endswith('/resume'):\n"
        f"    sys.exit(7) if {transport_failure!r} else print('403', end='')\n"
        "elif args[-1].endswith('/status'):\n"
        '    print(\'{"status":12,"streamlitVersion":"test"}\')\n'
        "else:\n"
        "    Path(args[args.index('-o') + 1]).write_text('<div id=\"root\"></div>')\n"
        "    print('200', end='')\n"
    )
    curl.chmod(0o755)
    sleep = tmp_path / "sleep"
    sleep.write_text("#!/bin/sh\necho 'unexpected polling' >&2\nexit 99\n")
    sleep.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/smoke_ui.sh"), "https://ui.example", "8"],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    expected = "wake request did not answer" if transport_failure else "HTTP 403"
    assert expected in result.stderr
    assert "scripts/smoke_ui_browser.py first" in result.stderr
    assert "unexpected polling" not in result.stderr
