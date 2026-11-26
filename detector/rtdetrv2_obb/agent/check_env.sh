#!/bin/bash

# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

set -e

FUNCTION_FILE_PATH="rtdetrv2_obb/func.py"

validate_access_token() {
    if [ -z "$CVAT_ACCESS_TOKEN" ]; then
        echo "Error: CVAT_ACCESS_TOKEN must be set."
        exit 1
    fi
}

resolve_base_url() {
    CVAT_BASE_URL="${CVAT_BASE_URL:-https://app.cvat.ai}"
}

resolve_org_slug() {
    ORG_SLUG_ARGS=()
    if [ -z "$ORG_SLUG" ]; then
        ORG_CURL_ARGS=(--data-urlencode "org=")
    else
        ORG_SLUG_ARGS=(--organization "$ORG_SLUG")
        ORG_CURL_ARGS=(--data-urlencode "org=$ORG_SLUG")
    fi
}

resolve_visibility() {
    if [ "$FUNCTION_VISIBILITY" = "public" ]; then
        VISIBILITY_ARGS=(--visibility public)
    else
        VISIBILITY_ARGS=(--visibility private)
    fi
}

resolve_model_params() {
    if ! [[ "$MODEL_CONFIG_PARAMS" =~ (^|[[:space:]])-p[[:space:]]+dataset=str:(dota|dior-r)($|[[:space:]]) ]]; then
        echo "Error: MODEL_CONFIG_PARAMS must include '-p dataset=str:dota' or '-p dataset=str:dior-r'."
        exit 1
    fi
}

resolve_function_name() {
    if [ -z "$FUNCTION_NAME" ]; then
        echo "Error: FUNCTION_NAME must be set."
        exit 1
    fi
}

resolve_cuda() {
    USE_CUDA_ARGS=()
    if [ "$USE_CUDA" = "true" ]; then
        USE_CUDA_ARGS=(-p device=str:cuda)
    fi
}

common_env() {
    validate_access_token
    resolve_base_url
    resolve_org_slug
    resolve_visibility
}
