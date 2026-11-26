# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cvat_sdk.auto_annotation as cvataa
from rtdetrv2_obb.classes import DIOR_R_CLASSES, DOTA_CLASSES
from rtdetrv2_obb.common import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_MODEL_SIZE,
    GROUP_DIOR_R,
    GROUP_DOTA_MULTI_SCALE,
    GROUP_DOTA_SINGLE_SCALE,
    RTDETRv2OBBFunction,
)


def create(
    dataset: str,
    model_size: str = DEFAULT_MODEL_SIZE,
    checkpoint_group: str | None = None,
    checkpoint: str | None = None,
    *,
    device: str = "cpu",
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> cvataa.DetectionFunction:
    if dataset == "dota":
        class_names = DOTA_CLASSES
        checkpoint_group = checkpoint_group or GROUP_DOTA_SINGLE_SCALE
        if checkpoint_group not in {GROUP_DOTA_SINGLE_SCALE, GROUP_DOTA_MULTI_SCALE}:
            raise ValueError(
                "DOTA checkpoint_group must be 'dota-single-scale' or 'dota-multi-scale'"
            )
    elif dataset == "dior-r":
        class_names = DIOR_R_CLASSES
        checkpoint_group = checkpoint_group or GROUP_DIOR_R
        if checkpoint_group != GROUP_DIOR_R:
            raise ValueError("DIOR-R checkpoint_group must be 'dior-r'")
    else:
        raise ValueError("dataset must be 'dota' or 'dior-r'")

    return RTDETRv2OBBFunction(
        class_names=class_names,
        checkpoint_group=checkpoint_group,
        model_size=model_size,
        checkpoint=checkpoint,
        device=device,
        conf_threshold=conf_threshold,
    )
