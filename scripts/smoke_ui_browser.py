"""Open the deployed UI in a browser and read what it says about the API.

    uv run --no-project --with playwright python scripts/smoke_ui_browser.py https://shelf-life-xxxx.streamlit.app
    (once: uv run --no-project --with playwright playwright install chromium)

`scripts/smoke_ui.sh` proves over HTTP that the app exists, that the host
reports it RUNNING, and that the Streamlit server answers. It cannot read the
page: Streamlit draws it over a WebSocket, so the one thing a visitor sees —
the paragraph the script wrote after calling the API — is invisible to curl.
That paragraph is where the deployment-specific failures show. On 2026-09-11
the public UI read "cannot reach the API at http://localhost:8000": the
`SHELF_LIFE_API` secret was not set on Community Cloud, every unit test was
green, and the HTTP checks above would have passed.

So this renders the page and asserts three things about the text of the app
frame, in the order they can fail:

  1. the app rendered at all — its own caption is on the page, not Streamlit's
     exception box or the host's error page;
  2. it did not say it cannot reach the API — which is the secret, the network
     path from the host to Render, or the API being down;
  3. what it did say about the API is one of the two honest states: the model
     details (a model is serving) or the "no model loaded" warning (the
     deliberate model-less deployment). Either proves the round trip.

Playwright is not a project dependency and must not become one: it is a
browser, and `requirements.txt` — which is exactly what Community Cloud
installs — has to stay a list of what the UI itself imports. Run it with
`uv run --with playwright`, which installs it beside the locked environment
for one command and nowhere else.

Exit codes: 0 rendered and reached the API · 1 a check failed · 2 usage or
playwright missing.
"""

from __future__ import annotations

import os
import sys
import time

#: The sentence `app/streamlit_app.py` puts under its title. Its presence is
#: how "the app rendered" is decided — the host's own pages never contain it.
CAPTION = "Will this job posting come off the board soon?"

#: The page opens on the board-ranking mode — the product (`design.md` §15) —
#: with or without a model: without one the buttons are disabled and the
#: workflow is still on screen, which is what a visitor to the public URL
#: should see while the panel accrues. Its absence would mean the deployed
#: UI is the single-posting demo again, or has stopped at a warning.
RANK_MODE = "Rank a board"

#: What the app says when the API is unreachable (`app/client.py`).
UNREACHABLE = "cannot reach the API"

#: The two honest states the app shows after a successful `/health`.
MODEL_LESS = "no model loaded"
MODEL_SERVING = "The model behind this form"

#: Streamlit's own failure surfaces.
CRASHED = ("Traceback", "Error running app", "Oh no.")


def app_text(page, deadline_s: float) -> str:
    """The text of the app frame once it has rendered past the spinner."""
    end = time.monotonic() + deadline_s
    last = ""
    while time.monotonic() < end:
        # The host page can be the sleeping page; a visitor would click.
        button = page.locator('[data-testid^="wakeup-button"]')
        if button.count():
            print("  the app was asleep; waking it")
            button.first.click()
            time.sleep(5)
        for frame in page.frames:
            if "/~/+/" not in frame.url:
                continue
            try:
                text = frame.inner_text("body").strip()
            except Exception:  # noqa: BLE001 - the frame is mid-navigation
                continue
            if text and "Waking the prediction service" not in text and text == last:
                return text
            last = text
        time.sleep(3)
    return last


def main(argv: list[str]) -> int:
    url = (argv[1] if len(argv) > 1 else os.environ.get("SHELF_LIFE_UI", "")).rstrip("/")
    if not url:
        print("usage: smoke_ui_browser.py <ui-url>   (or set SHELF_LIFE_UI)", file=sys.stderr)
        return 2
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed: uv run --with playwright ...", file=sys.stderr)
        return 2
    deadline_s = float(os.environ.get("UI_DEADLINE_SECONDS", "240"))

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(url + "/", wait_until="domcontentloaded", timeout=120_000)
        text = app_text(page, deadline_s)
        browser.close()

    one_line = " | ".join(line for line in text.splitlines() if line.strip())[:400]
    print(f"  rendered: {one_line}")

    if any(marker in text for marker in CRASHED):
        print("UI SMOKE FAIL: the app crashed while rendering", file=sys.stderr)
        return 1
    if CAPTION not in text:
        print("UI SMOKE FAIL: the app did not render its caption in time", file=sys.stderr)
        return 1
    if UNREACHABLE in text:
        print(
            "UI SMOKE FAIL: the UI cannot reach the API. Check the SHELF_LIFE_API secret on "
            "Community Cloud and that the API answers /health (docs/deploy.md §1).",
            file=sys.stderr,
        )
        return 1
    if RANK_MODE not in text:
        print("UI SMOKE FAIL: the page has no 'Rank a board' mode", file=sys.stderr)
        return 1
    if MODEL_LESS in text:
        print("ok  reached the API — no model loaded (the deliberate state); workflow shown")
    elif MODEL_SERVING in text:
        print("ok  reached the API — a model is serving; the page opens on 'Rank a board'")
    else:
        print(
            "UI SMOKE FAIL: rendered, but said nothing recognisable about the API", file=sys.stderr
        )
        return 1
    print(f"UI SMOKE PASS (browser): {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
