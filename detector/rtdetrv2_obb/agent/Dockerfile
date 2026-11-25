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

RUN mkdir -p /shared /app/rtdetrv2_obb/checkpoints/dior-r \
        /app/rtdetrv2_obb/checkpoints/dota-single-scale \
        /app/rtdetrv2_obb/checkpoints/dota-multi-scale && \
    chown -R agent:agent /shared /app /home/agent/.cache

COPY --chmod=755 detector/rtdetrv2_obb/clone_upstream.sh /app/rtdetrv2_obb/

RUN /app/rtdetrv2_obb/clone_upstream.sh && \
    chown -R agent:agent /app/rtdetrv2_obb/RiO-DETR

COPY --chmod=644 detector/rtdetrv2_obb/requirements.txt /app/rtdetrv2_obb/requirements.txt

RUN if [ "$USE_GPU" = "true" ] && [ "$TARGETARCH" = "amd64" ]; then \
        echo "Building GPU image" && pip install --no-cache-dir -r /app/rtdetrv2_obb/requirements.txt; \
    else \
        echo "Building CPU-only image" && \
        pip install --no-cache-dir --index-url=https://download.pytorch.org/whl/cpu torch torchvision && \
        pip install --no-cache-dir -r /app/rtdetrv2_obb/requirements.txt; \
    fi

ARG CVAT_VERSION

RUN test -n "$CVAT_VERSION" && \
    pip install --no-cache-dir "cvat-cli==$CVAT_VERSION" "cvat-sdk==$CVAT_VERSION" && \
    test "$(python -c 'from importlib.metadata import version; print(version("cvat-sdk"))')" = "$CVAT_VERSION" && \
    test "$(python -c 'from importlib.metadata import version; print(version("cvat-cli"))')" = "$CVAT_VERSION"

COPY --chmod=644 detector/rtdetrv2_obb/__init__.py detector/rtdetrv2_obb/common.py \
    detector/rtdetrv2_obb/classes.py detector/rtdetrv2_obb/func.py \
    detector/rtdetrv2_obb/upstream_loader.py /app/rtdetrv2_obb/
COPY --chmod=755 detector/rtdetrv2_obb/agent/function_registration.sh /app/
COPY --chmod=755 detector/rtdetrv2_obb/agent/function_deregistration.sh /app/
COPY --chmod=755 detector/rtdetrv2_obb/agent/check_env.sh /app/
COPY --chmod=755 detector/rtdetrv2_obb/agent/entrypoint.sh /app/

# v1.34.6 includes the fix for CVE-2025-68121.
ADD --chmod=+x https://dl.k8s.io/release/v1.34.6/bin/linux/${TARGETARCH}/kubectl /usr/local/bin/

USER agent

ENTRYPOINT ["./entrypoint.sh"]
