#!/bin/bash

# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

set -e

UPSTREAM_REPOSITORY="https://github.com/RicePasteM/RiO-DETR.git"
UPSTREAM_COMMIT="125c132d41946378e4f0e6067f0f8d28ade008fc"
UPSTREAM_DIR="${RIO_DETR_UPSTREAM_DIR:-$(dirname "$0")/RiO-DETR}"

if [ ! -d "$UPSTREAM_DIR/.git" ]; then
    git clone "$UPSTREAM_REPOSITORY" "$UPSTREAM_DIR"
fi

git -C "$UPSTREAM_DIR" fetch --depth 1 origin "$UPSTREAM_COMMIT"
git -C "$UPSTREAM_DIR" checkout --detach "$UPSTREAM_COMMIT"

ACTUAL_COMMIT=$(git -C "$UPSTREAM_DIR" rev-parse HEAD)
if [ "$ACTUAL_COMMIT" != "$UPSTREAM_COMMIT" ]; then
    echo "Error: expected RiO-DETR commit $UPSTREAM_COMMIT, got $ACTUAL_COMMIT"
    exit 1
fi

echo "Using RiO-DETR source at $UPSTREAM_DIR ($ACTUAL_COMMIT)"
