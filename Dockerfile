# The service, containerised. One artifact, two endpoints, no build-time model
# training — `python -m src.models.freeze` runs long before this.
#
# The model does not arrive by COPY. `models/` is derived output, is not
# committed, and is in `.dockerignore`, so the only way one enters this image is
# a verified fetch from a release tag:
#
#     docker build --build-arg ARTIFACT_TAG=artifact-2026-09-07 -t shelf-life .
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
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, in their own layer. They change when `pyproject.toml`
# does, which is rarely; the source changes constantly, and copying it first
# would reinstall scikit-learn and XGBoost on every edit.
COPY pyproject.toml README.md ./
COPY src ./src
COPY api ./api
RUN pip install --no-cache-dir ".[api]" \
    # XGBoost's Linux wheel depends on the CUDA runtime, which is 291MB of GPU
    # libraries this service will never call: it scores one row at a time on a
    # free tier that has no GPU. Removing them takes site-packages from 913MB to
    # 622MB, and the pinned prediction is byte-identical afterwards. Nothing
    # imports them unless a booster is asked for `device="cuda"`.
    && pip list --format=freeze | grep -i "^nvidia" | cut -d= -f1 | xargs -r pip uninstall -y

# Then everything else the image is allowed to have. `.dockerignore` decides
# what "everything else" means, and it does not mean `models/`.
COPY . .

# The model, by tag, or not at all. Empty by default so a plain `docker build`
# still works and still produces the honest no-artifact container.
ARG ARTIFACT_REPO=professor3333/shelf-life
ARG ARTIFACT_TAG=""

# Two checks, not one, and they prove different things. The fetch verifies the
# bytes against the checksums published with the release. The load then proves
# the file is a fitted end-to-end pipeline *in the environment that will serve
# it* — which is the check no checksum can make, and the one that would have
# caught a booster written by a different xgboost than the image installs.
# Both run at build time, so a bad artifact fails the build rather than the
# stranger's first request.
RUN if [ -n "${ARTIFACT_TAG}" ]; then \
        python -m src.inference.fetch --repo "${ARTIFACT_REPO}" --tag "${ARTIFACT_TAG}" --into models && \
        python -c "from src.inference.artifact import load; a = load(); \
print(f'artifact ok: {a.metadata.run_name}, fitted on {a.metadata.fitted_on}, threshold {a.metadata.threshold}')"; \
    else \
        echo "no ARTIFACT_TAG: building the no-artifact image (/health will report model_loaded: false)"; \
    fi

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
