# The service, containerised. One artifact, two endpoints, no build-time model
# training — `python -m src.models.freeze` runs long before this.
#
# The model does not arrive by COPY. `models/` is derived output, is not
# committed, and is in `.dockerignore`, so the only way one enters this image is
# a verified fetch from a release tag. The tag comes from the committed
# `MODEL_TAG` file, which is what makes "which model is serving?" answerable
# from git history rather than from a dashboard (`docs/design.md` §7f). A build
# argument overrides it, for building a specific version locally:
#
#     docker build -t shelf-life .                                   # MODEL_TAG
#     docker build --build-arg ARTIFACT_TAG=artifact-2026-09-07 .    # override
#
# That is `docs/design.md` §7a: the served model is a version, not a file that
# happened to be on a laptop. Locally, mount one instead of baking it —
# `docker run -v "$PWD/models:/app/models:ro"`.
#
# Without a tag the image still builds, deliberately: the container starts,
# `/health` answers 200 with `model_loaded: false`, and `/predict` returns 503
# with the command that fixes it. A container that refuses to boot because a
# file is missing turns a one-line diagnosis into a log-reading exercise.

FROM python:3.12-slim

# Bytecode files and buffered stdout both cost more than they are worth in a
# container: the first is written to a layer nobody reads, the second hides the
# logs of a process that just died.
#
# The uv settings make the install a function of `uv.lock` and nothing else:
# the environment lands at a fixed path, from the image's own interpreter (no
# managed-Python download), with no cache left in a layer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1

# The installer, pinned to the version that wrote the lock. Installing it with
# pip would fetch whatever uv was newest that day, which is the one unpinned
# thing in an image whose point is that nothing is.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv

WORKDIR /app

# Dependencies first, in their own layer, from the lock. `--locked` refuses if
# `uv.lock` has fallen behind `pyproject.toml`, so an edited dependency list
# with a stale lock fails the build rather than quietly installing something
# the lock does not describe. `--no-install-project` keeps the source out of
# this layer: the lock changes rarely, the source constantly, and copying it
# first would reinstall scikit-learn and XGBoost on every edit.
#
# Only the `api` extra. The image serves; it does not train, track, plot or
# render a form, and each of those extras is a dependency tree this 0.1-CPU
# instance would pay for on every cold start.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --extra api

# The environment first on PATH, so `python` and `uvicorn` below — and in the
# healthcheck and the command — mean the locked ones.
ENV PATH="/app/.venv/bin:${PATH}"

# Then everything else the image is allowed to have. `.dockerignore` decides
# what "everything else" means, and it does not mean `models/`. The second sync
# installs the project itself, as a built wheel rather than an editable link,
# against the already-populated environment.
#
# The uninstall comes *after* the last sync, because a sync makes the
# environment match the lock and would put back anything removed before it —
# the first draft removed the CUDA libraries in the layer above and the second
# sync reinstalled them. XGBoost's Linux wheel depends on the CUDA runtime,
# hundreds of MB of GPU libraries this service will never call: it scores one
# row at a time on a free tier that has no GPU. Nothing imports them unless a
# booster is asked for `device="cuda"`, and the pinned prediction is
# byte-identical without them. The lock still names them, deliberately: it has
# to reproduce the freeze's environment elsewhere, and a resolution edited by
# hand is no longer the lock.
COPY . .
RUN uv sync --locked --no-editable --extra api \
    && uv pip list --format=freeze | grep -i "^nvidia" | cut -d= -f1 | xargs -r uv pip uninstall

# The model, by tag, or not at all. Empty by default: the tag is normally read
# from `MODEL_TAG` below, and a plain `docker build` with neither still works and
# still produces the honest no-artifact container.
ARG ARTIFACT_REPO=professor3333/shelf-life
ARG ARTIFACT_TAG=""

# Two checks, not one, and they prove different things. The fetch verifies the
# bytes against the checksums published with the release. The load then proves
# the file is a fitted end-to-end pipeline *in the environment that will serve
# it* — which is the check no checksum can make, and the one that would have
# caught a booster written by a different xgboost than the image installs.
# Both run at build time, so a bad artifact fails the build rather than the
# stranger's first request.
#
# The load is strict here where it is lenient elsewhere: `load()` *warns* when
# the artifact's recorded scikit-learn, xgboost or joblib differ from the
# installed ones, because a laptop loading an old artifact to look at it should
# not be refused. An image is different — it exists to serve that artifact, and
# "the environment drifted since the freeze" is exactly the failure a build
# should stop. The warning is promoted to an error, by message, so nothing else
# a library says at import time can fail the build in its place. Since the
# environment comes from the lock and the freeze recorded the lock's hash, a
# mismatch here means the lock moved after the freeze: re-freeze, or do not
# bump the library the artifact was written with.
#
# The build argument wins if given; otherwise the committed file decides. Blank
# lines and `#` comments are stripped from it, so `MODEL_TAG` can explain itself
# and a file containing only its own explanation reads as "no model yet" rather
# than as a tag named `#`.
#
# The last line records what was *actually* used, so `/health` can report the
# tag this build fetched rather than the tag the repository currently intends.
# `ENV` cannot take a value computed by `RUN`, so it is a file, and
# `api/main.py` reads it. (No comments inside the RUN itself: the Dockerfile
# parser strips comment lines out of a backslash continuation.)
RUN TAG="${ARTIFACT_TAG}"; \
    if [ -z "${TAG}" ] && [ -f MODEL_TAG ]; then \
        TAG=$(sed -e 's/#.*//' -e 's/[[:space:]]//g' MODEL_TAG | grep -v '^$' | head -n 1); \
    fi; \
    if [ -n "${TAG}" ]; then \
        echo "artifact tag: ${TAG}" && \
        python -m src.inference.fetch --repo "${ARTIFACT_REPO}" --tag "${TAG}" --into models && \
        python -c "import warnings; warnings.filterwarnings('error', message='artifact was written with'); \
from src.inference.artifact import load; from src.models.provenance import lock_sha256; a = load(); \
print(f'artifact ok: {a.metadata.run_name}, fitted on {a.metadata.fitted_on}, threshold {a.metadata.threshold}'); \
print(f'libraries: {a.metadata.versions}'); \
same = a.metadata.lock_sha256 == lock_sha256(); \
print('lock: ' + ('the one the artifact was frozen against' if same else 'moved since the freeze; the library versions above still agree'))"; \
    else \
        echo "no artifact tag: building the no-artifact image (/health will report model_loaded: false)"; \
    fi; \
    printf '%s' "${TAG}" > /app/ARTIFACT_TAG

# Not root. The process needs to read one artifact and answer HTTP; it has no
# business being able to write to its own image.
RUN useradd --create-home --uid 10001 shelflife && chown -R shelflife:shelflife /app
USER shelflife

# Free tiers inject $PORT and expect the process to honour it. The default keeps
# `docker run -p 8000:8000` working without one.
ENV PORT=8000
EXPOSE 8000

# Uses the app's own endpoint rather than a TCP probe, so "the port is open" and
# "the model loaded" stay distinguishable — which is the whole point of /health
# reporting them separately.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen(f\"http://localhost:{os.environ['PORT']}/health\", timeout=4).status == 200 else 1)"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
