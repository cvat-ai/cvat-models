# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cvat_sdk.auto_annotation as cvataa
from rio_detr.common import (DEFAULT_CONF_THRESHOLD, DEFAULT_MODEL_SIZE,
                             GROUP_DIOR_R, RioDetrFunction)
from rio_detr.dior_r.dior_r_classes import DIOR_R_CLASSES


def create(
    model_size: str = DEFAULT_MODEL_SIZE,
    checkpoint_group: str = GROUP_DIOR_R,
    checkpoint: str | None = None,
    *,
    device: str = "cpu",
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> cvataa.DetectionFunction:
    if checkpoint_group != GROUP_DIOR_R:
        raise ValueError("DIOR-R checkpoint_group must be 'dior-r'")
    return RioDetrFunction(
        class_names=DIOR_R_CLASSES,
        checkpoint_group=checkpoint_group,
        model_size=model_size,
        checkpoint=checkpoint,
        device=device,
        conf_threshold=conf_threshold,
    )
