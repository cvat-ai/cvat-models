#!/bin/bash

# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

set -e
source "$(dirname "$0")/check_env.sh"

common_env
resolve_model_params
resolve_cuda

if [ -f /shared/FUNCTION_ID ]; then
    FUNCTION_ID=$(cat /shared/FUNCTION_ID)
fi
if [ -z "$FUNCTION_ID" ]; then
    echo "Error: FUNCTION_ID is not set and /shared/FUNCTION_ID does not exist."
    exit 1
fi

/app/rtdetrv2_obb/clone_upstream.sh
exec cvat-cli --server-host "$CVAT_BASE_URL" "${ORG_SLUG_ARGS[@]}" function run-agent \
    "$FUNCTION_ID" --function-file="rtdetrv2_obb/dota/func.py" $MODEL_CONFIG_PARAMS "${USE_CUDA_ARGS[@]}"
