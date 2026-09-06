"""The deploy path, checked without deploying anything.

Nothing here talks to Google, GitHub or Hugging Face. What it checks is the
class of mistake that a deploy does not report: a rename in one file that
silently changes what another file does.

**The one worth having is `test_the_workflow_passes_build_args_the_dockerfile_declares`.**
Docker ignores a `--build-arg` naming an `ARG` that does not exist — it warns,
and the build succeeds. The Dockerfile's model fetch is guarded by
`if [ -n "${ARTIFACT_TAG}" ]`, so an unrecognised argument does not fail the
build: it produces the *no-artifact* image, which starts, passes its health
check, reports `model_loaded: false` and returns 503 to every caller. Renaming
that `ARG` would therefore deploy an empty service and no step before the smoke
test would notice.
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
WORKFLOW = ROOT / ".github" / "workflows" / "deploy.yml"
DOCKERFILE = ROOT / "Dockerfile"
RUNBOOK = ROOT / "docs" / "deploy.md"
SPACE = ROOT / "deploy" / "space"


@pytest.fixture(scope="module")
def workflow_text() -> str:
    return WORKFLOW.read_text()


@pytest.fixture(scope="module")
def workflow(workflow_text: str) -> dict:
    return yaml.safe_load(workflow_text)


def test_the_workflow_passes_build_args_the_dockerfile_declares(workflow_text: str) -> None:
    passed = set(re.findall(r'--build-arg\s+"?([A-Z_]+)=', workflow_text))
    declared = set(re.findall(r"^ARG\s+([A-Z_]+)", DOCKERFILE.read_text(), re.MULTILINE))

    assert passed, "the deploy workflow passes no build args; the image would have no model"
    unknown = passed - declared
    assert not unknown, (
        f"the workflow passes {sorted(unknown)}, which the Dockerfile does not declare. "
        "Docker only warns about that, so the build would succeed and deploy the "
        "no-artifact image — a service that answers 503 to everyone."
    )
    assert "ARTIFACT_TAG" in passed, "without ARTIFACT_TAG the deployed image has no model in it"


def test_the_tag_trigger_matches_the_tags_the_runbook_tells_you_to_push(workflow) -> None:
    # `on:` is YAML's `True`. Ask for both rather than pretending to be surprised.
    triggers = workflow.get("on") or workflow.get(True)
    patterns = triggers["push"]["tags"]

    documented = set(re.findall(r"artifact-[\w<>{}$()+%\-]+", RUNBOOK.read_text()))
    assert documented, "the runbook documents no artifact tag at all"
    assert any(
        pattern.rstrip("*") and tag.startswith(pattern.rstrip("*"))
        for pattern in patterns
        for tag in documented
    ), f"the workflow triggers on {patterns}, which no tag in the runbook would match"


def test_preflight_checks_every_variable_the_workflow_later_uses(workflow_text: str) -> None:
    """A variable used but not preflighted fails four minutes later, in gcloud's words."""
    used = set(re.findall(r"vars\.([A-Z_]+)", workflow_text))
    preflight = workflow_text.split("Preflight — is this repository configured")[1]
    checked = set(re.findall(r"\b(GCP_[A-Z_]+)\b", preflight.split("- name:")[0]))

    missing = used - checked
    assert not missing, (
        f"{sorted(missing)} is used by the deploy but not checked by the preflight, so a "
        "repository missing it fails deep inside gcloud instead of in the first ten seconds"
    )


def test_the_workflow_smoke_tests_what_it_deployed(workflow_text: str) -> None:
    """`gcloud run deploy` succeeding is not evidence that a model is serving."""
    assert "scripts/smoke.sh" in workflow_text, (
        "the workflow deploys without smoke-testing; a revision that answers 503 to "
        "every caller is a successful deploy"
    )
    smoke = ROOT / "scripts" / "smoke.sh"
    assert smoke.exists() and os.access(smoke, os.X_OK), f"{smoke} is missing or not executable"


def test_the_smoke_test_refuses_to_pass_a_synthetic_model() -> None:
    """The rehearsal model must not reach a public URL unannounced.

    Every component in this project is exercised on a synthetic panel whose label
    is drawn independently of every feature, so its predictions are noise. That
    is the right thing to build against and the wrong thing to serve.
    """
    smoke = (ROOT / "scripts" / "smoke.sh").read_text()
    assert "synthetic" in smoke and "ALLOW_SYNTHETIC" in smoke


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


def _space_requirements() -> set[str]:
    """The distribution names the Space installs. Comments are prose, not dependencies."""
    return {
        re.split(r"[<>=!~ ]", line, maxsplit=1)[0].strip().lower()
        for line in (SPACE / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_the_space_requirements_cover_everything_the_ui_imports() -> None:
    """A missing line here is a Space that crashes on its first page load."""
    required = _third_party_imports(ROOT / "app")
    declared = _space_requirements()
    missing = {name for name in required if name.lower() not in declared}
    assert not missing, (
        f"app/ imports {sorted(missing)}, which deploy/space/requirements.txt does not "
        "install. The Space installs that file and nothing else."
    )


def test_the_space_cannot_install_a_model() -> None:
    """UI ≠ API ≠ model. On the Space that is enforced by what is not installed."""
    declared = _space_requirements()
    for forbidden in ("scikit-learn", "sklearn", "xgboost", "joblib"):
        assert forbidden not in declared, (
            f"deploy/space/requirements.txt installs {forbidden}. The UI calls the API over "
            "HTTP; a UI that can load a model is a second copy of it, silently different."
        )


def test_the_space_points_at_a_file_that_exists() -> None:
    front_matter = (SPACE / "README.md").read_text().split("---")[1]
    app_file = yaml.safe_load(front_matter)["app_file"]
    assert (ROOT / app_file).exists(), f"the Space would launch {app_file}, which is not here"


def test_every_module_the_runbook_tells_you_to_run_exists() -> None:
    """A runbook that names a module nobody can import is worse than no runbook."""
    for module in sorted(set(re.findall(r"python -m ([\w.]+)", RUNBOOK.read_text()))):
        assert importlib.util.find_spec(module) is not None, (
            f"docs/deploy.md says to run `python -m {module}`, which does not exist"
        )
