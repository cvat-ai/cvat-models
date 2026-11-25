# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    jq \
    && rm -rf /var/lib/apt/lists/*

ARG USE_GPU=false
ARG TARGETARCH

RUN groupadd --gid 1000 agent && \
    useradd --uid 1000 --gid 1000 --create-home agent

WORKDIR /app

RUN mkdir -p /shared /app/rio_detr/checkpoints/dior-r \
        /app/rio_detr/checkpoints/dota-single-scale \
        /app/rio_detr/checkpoints/dota-multi-scale \
        /home/agent/.cache/cvat-models/rio-detr && \
    chown -R agent:agent /shared /app /home/agent/.cache

COPY --chmod=644 detector/rio_detr/requirements.txt /app/rio_detr/requirements.txt

RUN if [ "$USE_GPU" = "true" ] && [ "$TARGETARCH" = "amd64" ]; then \
        echo "Building GPU image" && pip install --no-cache-dir -r /app/rio_detr/requirements.txt; \
    else \
        echo "Building CPU-only image" && \
        pip install --no-cache-dir --index-url=https://download.pytorch.org/whl/cpu torch torchvision && \
        pip install --no-cache-dir -r /app/rio_detr/requirements.txt; \
    fi

ARG CVAT_VERSION

RUN test -n "$CVAT_VERSION" && \
    pip install --no-cache-dir "cvat-cli==$CVAT_VERSION" "cvat-sdk==$CVAT_VERSION" && \
    test "$(python -c 'from importlib.metadata import version; print(version("cvat-sdk"))')" = "$CVAT_VERSION" && \
    test "$(python -c 'from importlib.metadata import version; print(version("cvat-cli"))')" = "$CVAT_VERSION"

COPY --chmod=644 detector/rio_detr/__init__.py detector/rio_detr/common.py \
    detector/rio_detr/upstream_loader.py /app/rio_detr/
COPY --chmod=755 detector/rio_detr/clone_upstream.sh /app/rio_detr/
RUN mkdir -p /app/rio_detr/dota
COPY --chmod=644 detector/rio_detr/dota/__init__.py detector/rio_detr/dota/func.py \
    detector/rio_detr/dota/dota_classes.py /app/rio_detr/dota/
COPY --chmod=755 detector/rio_detr/dota/agent/function_registration.sh /app/
COPY --chmod=755 detector/rio_detr/dota/agent/function_deregistration.sh /app/
COPY --chmod=755 detector/rio_detr/dota/agent/check_env.sh /app/
COPY --chmod=755 detector/rio_detr/dota/agent/entrypoint.sh /app/

# v1.34.6 includes the fix for CVE-2025-68121.
ADD --chmod=+x https://dl.k8s.io/release/v1.34.6/bin/linux/${TARGETARCH}/kubectl /usr/local/bin/

USER agent

ENV RIO_DETR_UPSTREAM_DIR=/home/agent/.cache/cvat-models/rio-detr/RiO-DETR

ENTRYPOINT ["./entrypoint.sh"]
