# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cvat_sdk.auto_annotation as cvataa
from rtdetrv2_obb.common import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_MODEL_SIZE,
    GROUP_DOTA_MULTI_SCALE,
    GROUP_DOTA_SINGLE_SCALE,
    RTDETRv2OBBFunction,
)
from rtdetrv2_obb.dota.dota_classes import DOTA_CLASSES


def create(
    model_size: str = DEFAULT_MODEL_SIZE,
    checkpoint_group: str = GROUP_DOTA_SINGLE_SCALE,
    checkpoint: str | None = None,
    *,
    device: str = "cpu",
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> cvataa.DetectionFunction:
    if checkpoint_group not in {GROUP_DOTA_SINGLE_SCALE, GROUP_DOTA_MULTI_SCALE}:
        raise ValueError(
            "DOTA checkpoint_group must be 'dota-single-scale' or 'dota-multi-scale'"
        )
    return RTDETRv2OBBFunction(
        class_names=DOTA_CLASSES,
        checkpoint_group=checkpoint_group,
        model_size=model_size,
        checkpoint=checkpoint,
        device=device,
        conf_threshold=conf_threshold,
    )
