#!/bin/bash

# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

set -e
source "$(dirname "$0")/check_env.sh"

common_env
if [ -z "$KUBERNETES_SERVICE_HOST" ] && [ -f /shared/FUNCTION_ID ]; then
    FUNCTION_ID=$(cat /shared/FUNCTION_ID)
fi
if [ -z "$FUNCTION_ID" ]; then
    echo "No function ID was found; nothing to deregister."
    exit 0
fi
if [ -n "$KUBERNETES_SERVICE_HOST" ]; then
    kubectl delete configmap "$CONFIGMAP_NAME" --namespace "$NAMESPACE"
fi
exec cvat-cli --server-host "$CVAT_BASE_URL" "${ORG_SLUG_ARGS[@]}" function delete "$FUNCTION_ID"
