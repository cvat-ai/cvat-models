# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: AGPL-3.0-or-later

FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    jq \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ARG USE_GPU=false
ARG TARGETARCH
ARG CVAT_VERSION

RUN groupadd --gid 1000 agent && \
    useradd --uid 1000 --gid 1000 --create-home agent

WORKDIR /app

RUN mkdir /shared && chown agent:agent /shared && chown agent:agent /app

COPY --chmod=755 detector/yolo/requirements.txt .

RUN test -n "$CVAT_VERSION" && \
    if [ "$USE_GPU" = "true" ] && [ "$TARGETARCH" = "amd64" ]; then \
        echo "Building GPU image" && pip install --no-cache-dir "cvat-cli==$CVAT_VERSION" "cvat-sdk==$CVAT_VERSION" -r requirements.txt; \
    else \
        echo "Building CPU only image" && pip install --no-cache-dir "cvat-cli==$CVAT_VERSION" "cvat-sdk==$CVAT_VERSION" -r requirements.txt --extra-index-url=https://download.pytorch.org/whl/cpu; \
    fi && \
    test "$(python -c 'from importlib.metadata import version; print(version("cvat-sdk"))')" = "$CVAT_VERSION" && \
    test "$(python -c 'from importlib.metadata import version; print(version("cvat-cli"))')" = "$CVAT_VERSION"

COPY --chmod=755 detector/yolo/func.py ./
COPY --chmod=755 detector/yolo/agent/function_registration.sh .
COPY --chmod=755 detector/yolo/agent/function_deregistration.sh .
COPY --chmod=755 detector/yolo/agent/check_env.sh .
COPY --chmod=755 detector/yolo/agent/entrypoint.sh .
COPY --chmod=755 detector/yolo/agent/keypoints.txt .

#v1.34.5 fixes CVE-2025-68121
ADD --chmod=+x https://dl.k8s.io/release/v1.34.6/bin/linux/${TARGETARCH}/kubectl /usr/local/bin/

USER agent

ENTRYPOINT ["./entrypoint.sh"]
