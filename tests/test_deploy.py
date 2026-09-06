"""The deploy path, checked without deploying anything.

Nothing here talks to Render, GitHub or Streamlit. What it checks is the class of
mistake that a deploy does not report: a rename in one file that silently changes
what another file does, and a configuration change that quietly costs money.

**`test_the_service_plan_is_free` is the one with a bill attached.** Every other
failure here is an outage; that one is a charge. The constraint this deployment
is built to (`docs/design.md` §7) is not "cheap" but "$0 with no payment method
on file", and a plan named in a YAML file is exactly the kind of thing that gets
bumped during debugging and left.

**`test_the_workflow_reads_model_tag_the_same_way_the_dockerfile_does` is the
one with an invisible failure.** The tag is extracted by an identical shell
pipeline in two places. If they drift, the workflow waits for a release the image
was never built with, times out, and reports a deployment failure that did not
happen.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "verify-deployment.yml"
DOCKERFILE = ROOT / "Dockerfile"
RENDER = ROOT / "render.yaml"
RUNBOOK = ROOT / "docs" / "deploy.md"
REQUIREMENTS = ROOT / "requirements.txt"

#: The pipeline that turns the `MODEL_TAG` file into a bare tag. Comments and
#: whitespace out, first surviving line wins.
TAG_PIPELINE = "sed -e 's/#.*//' -e 's/[[:space:]]//g' MODEL_TAG | grep -v '^$' | head -n 1"


@pytest.fixture(scope="module")
def workflow_text() -> str:
    return WORKFLOW.read_text()


@pytest.fixture(scope="module")
def render() -> dict:
    return yaml.safe_load(RENDER.read_text())["services"][0]


# --- the ones about money ----------------------------------------------------


def test_the_service_plan_is_free(render: dict) -> None:
    """$0 is the first requirement, and this file is where it would be lost."""
    assert render["plan"] == "free", (
        f"render.yaml asks for the {render['plan']!r} plan. The deployment constraint is $0 "
        "with no payment method on file (docs/design.md §7); a paid plan is a bill, not a "
        "performance tweak."
    )


def test_nothing_in_the_blueprint_scales_beyond_the_free_instance(render: dict) -> None:
    """Free services are single-instance. Asking for more is asking for a plan change."""
    for field in ("numInstances", "scaling"):
        assert field not in render, (
            f"render.yaml sets {field!r}, which the free plan does not support and which "
            "Render would satisfy by requiring a paid plan."
        )


# --- the ones about a deploy that silently did not happen --------------------


def test_the_workflow_reads_model_tag_the_same_way_the_dockerfile_does(workflow_text: str) -> None:
    """Two copies of one pipeline. Drift here fails a deploy that worked."""
    assert TAG_PIPELINE in DOCKERFILE.read_text(), (
        "the Dockerfile no longer extracts the tag from MODEL_TAG the documented way; "
        "if that is deliberate, update TAG_PIPELINE here and the workflow together"
    )
    assert TAG_PIPELINE in workflow_text, (
        "the verify workflow extracts the release tag differently from the Dockerfile. "
        "They must agree: the workflow waits for the tag the image was built with, and a "
        "mismatch reports a deployment failure that did not occur."
    )


def test_the_dockerfile_records_the_tag_it_actually_fetched(workflow_text: str) -> None:
    """`/health` must report the release the build used, not the one the repo intends."""
    dockerfile = DOCKERFILE.read_text()
    assert "> /app/ARTIFACT_TAG" in dockerfile, (
        "the image no longer records which release it fetched, so /health cannot report it "
        "and the verify workflow cannot tell a new deployment from the one it replaced"
    )
    assert "artifact_tag" in workflow_text or "await_release.sh" in workflow_text


def test_the_build_arg_override_still_matches_the_dockerfile(workflow_text: str) -> None:
    """Docker only *warns* about a --build-arg naming an ARG that does not exist."""
    declared = set(re.findall(r"^ARG\s+([A-Z_]+)", DOCKERFILE.read_text(), re.MULTILINE))
    assert {"ARTIFACT_TAG", "ARTIFACT_REPO"} <= declared, (
        "the local override documented in the Dockerfile header and docs/deploy.md relies on "
        f"these build args; the Dockerfile declares {sorted(declared)}"
    )
    for passed in set(re.findall(r'--build-arg\s+"?([A-Z_]+)=', workflow_text)):
        assert passed in declared, (
            f"the workflow passes --build-arg {passed}, which the Dockerfile does not declare. "
            "Docker only warns, so the build would succeed and produce the no-artifact image."
        )


def test_the_workflow_verifies_rather_than_assumes(workflow_text: str) -> None:
    """A deploy check that cannot fail is decoration."""
    for script in ("scripts/await_release.sh", "scripts/smoke.sh"):
        assert script in workflow_text, f"the workflow never runs {script}"
        path = ROOT / script
        assert path.exists() and os.access(path, os.X_OK), f"{path} is missing or not executable"


def test_the_health_check_path_is_one_the_api_serves(render: dict) -> None:
    main = (ROOT / "api" / "main.py").read_text()
    served = set(re.findall(r'@app\.(?:get|post)\("([^"]+)"', main))
    assert render["healthCheckPath"] in served, (
        f"render.yaml health-checks {render['healthCheckPath']}, which the API does not serve; "
        f"it serves {sorted(served)}"
    )


def test_the_smoke_test_refuses_a_synthetic_model() -> None:
    """The rehearsal model must not reach a public URL unannounced.

    Every component in this project is exercised on a synthetic panel whose label
    is drawn independently of every feature, so its predictions are noise. That
    is the right thing to build against and the wrong thing to serve.
    """
    smoke = (ROOT / "scripts" / "smoke.sh").read_text()
    assert "synthetic" in smoke and "ALLOW_SYNTHETIC" in smoke


# --- the ones about the UI's boundary ----------------------------------------


def _third_party_imports(directory: Path) -> set[str]:
    """Top-level module names imported by a package, minus the standard library."""
    found: set[str] = set()
    for path in sorted(directory.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    stdlib = set(getattr(__import__("sys"), "stdlib_module_names", ()))
    return {name for name in found if name not in stdlib and name != "app"}


def _ui_requirements() -> set[str]:
    """What Streamlit Community Cloud installs. Comments are prose, not dependencies."""
    return {
        re.split(r"[<>=!~ ]", line, maxsplit=1)[0].strip().lower()
        for line in REQUIREMENTS.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_the_ui_requirements_cover_everything_the_ui_imports() -> None:
    """A missing line here is a UI that crashes on its first page load."""
    missing = {n for n in _third_party_imports(ROOT / "app") if n.lower() not in _ui_requirements()}
    assert not missing, (
        f"app/ imports {sorted(missing)}, which requirements.txt does not install. "
        "Streamlit Community Cloud installs that file and nothing else."
    )


def test_the_ui_cannot_install_a_model() -> None:
    """UI ≠ API ≠ model, enforced by what the UI's host is never given.

    Weaker than the previous arrangement, where `src/` was not deployed alongside
    the UI at all and the boundary held by physics. Community Cloud checks out
    the whole repository, so this list and `tests/test_app.py` are what remain.
    `docs/design.md` §7c records that as a downgrade rather than a preference.
    """
    declared = _ui_requirements()
    for forbidden in ("scikit-learn", "sklearn", "xgboost", "joblib"):
        assert forbidden not in declared, (
            f"requirements.txt installs {forbidden}, so the UI could load the artifact "
            "directly. A UI that can load a model is a second copy of it, silently different."
        )


# --- the one about which model is answering ----------------------------------


def test_health_reports_the_release_only_when_the_build_recorded_one(monkeypatch, tmp_path) -> None:
    """Intent and fact are different questions, and /health answers the second.

    `MODEL_TAG` says which release the repository *wants* deployed. The file the
    Dockerfile writes says which one the build actually fetched. Reporting the
    first as if it were the second would make the verify workflow unable to
    detect the one thing it exists to detect: a push whose image never got built.
    """
    from api import main

    monkeypatch.delenv(main.ARTIFACT_TAG_ENV, raising=False)

    recorded = tmp_path / "ARTIFACT_TAG"
    monkeypatch.setattr(main, "ARTIFACT_TAG_FILE", recorded)
    assert main.artifact_tag() is None, "no recorded tag must read as None, not as an empty string"

    recorded.write_text("artifact-2026-09-08\n")
    assert main.artifact_tag() == "artifact-2026-09-08"

    recorded.write_text("")
    assert main.artifact_tag() is None, "the no-artifact image writes an empty file, not a tag"

    monkeypatch.setenv(main.ARTIFACT_TAG_ENV, "artifact-override")
    assert main.artifact_tag() == "artifact-override", "the environment must win over the file"


# --- the one about documentation that has rotted -----------------------------


def test_every_module_the_runbook_tells_you_to_run_exists() -> None:
    """A runbook that names a module nobody can import is worse than no runbook."""
    for module in sorted(set(re.findall(r"python -m ([\w.]+)", RUNBOOK.read_text()))):
        assert importlib.util.find_spec(module) is not None, (
            f"docs/deploy.md says to run `python -m {module}`, which does not exist"
        )
