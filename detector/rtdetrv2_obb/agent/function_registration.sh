#!/bin/bash

# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

set -e
source "$(dirname "$0")/check_env.sh"

common_env
resolve_model_params
resolve_function_name

USERNAME=$(curl "$CVAT_BASE_URL"/api/users/self -s --fail --oauth2-bearer "$CVAT_ACCESS_TOKEN" | jq -r '.username')
if [ -z "$USERNAME" ] || [ "$USERNAME" = "null" ]; then
    echo "Error: unable to retrieve the CVAT username."
    exit 1
fi

FILTER=$(jq -nc --arg name "$FUNCTION_NAME" --arg owner "$USERNAME" \
  '{"and":[{"==":[{"var":"name"},$name]},{"==":[{"var":"owner"},$owner]}]}')

if FUNCTION_ID=$(curl --get --fail --data-urlencode "filter=$FILTER" "${ORG_CURL_ARGS[@]}" \
    "$CVAT_BASE_URL"/api/functions -s --oauth2-bearer "$CVAT_ACCESS_TOKEN" | jq -r '.results[0].id') && \
    [ "$FUNCTION_ID" != "null" ]; then
    echo "Found existing function with ID: $FUNCTION_ID"
else
    RAW_OUTPUT=$(cvat-cli --server-host "$CVAT_BASE_URL" "${ORG_SLUG_ARGS[@]}" function create-native \
        "$FUNCTION_NAME" "${VISIBILITY_ARGS[@]}" --function-file="$FUNCTION_FILE_PATH" $MODEL_CONFIG_PARAMS)
    FUNCTION_ID=$(echo "$RAW_OUTPUT" | tail -1 | tr -d '[:space:]')
    if ! [[ $FUNCTION_ID =~ ^[0-9]+$ ]]; then
        echo "cvat-cli function create-native failed:"
        echo "$RAW_OUTPUT"
        exit 1
    fi
fi
echo "$FUNCTION_ID" > /shared/FUNCTION_ID

if [ -z "$KUBERNETES_SERVICE_HOST" ]; then
    exit 0
fi
if kubectl -n "$NAMESPACE" get cm "$CONFIGMAP_NAME" >/dev/null 2>&1; then
    exec kubectl -n "$NAMESPACE" patch configmap "$CONFIGMAP_NAME" \
        -p "$(jq -n --arg func_id "$FUNCTION_ID" '{"data":{"FUNCTION_ID": $func_id}}')"
fi
exec kubectl create configmap "$CONFIGMAP_NAME" --namespace "$NAMESPACE" \
    --from-literal=FUNCTION_ID="$FUNCTION_ID"
