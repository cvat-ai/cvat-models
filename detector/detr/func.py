# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import abc
import json
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import cv2
import cvat_sdk.auto_annotation as cvataa
import cvat_sdk.models as models
import numpy as np
from cvat_sdk.masks import encode_mask
from PIL import Image
from rfdetr import RFDETR

# Only nondeprecated models distributed under the Apache-2.0 license are supported.
SUPPORTED_MODELS = (
    "RFDETRNano",
    "RFDETRSmall",
    "RFDETRMedium",
    "RFDETRLarge",
    "RFDETRSegNano",
    "RFDETRSegSmall",
    "RFDETRSegMedium",
    "RFDETRSegLarge",
    "RFDETRSegXLarge",
    "RFDETRSeg2XLarge",
    "RFDETRKeypointPreview",
)

DEFAULT_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


class DetrFunction(abc.ABC):
    def __init__(self, model: RFDETR, *, class_names: list[str] | None = None) -> None:
        self._model = model
        names = list(model.class_names) if class_names is None else class_names
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Model must provide nonempty class names")
        if len(set(names)) != len(names):
            raise ValueError("Model class names must be unique")

        # these are function-local IDs, not sparse COCO category IDs.
        # predictions are mapped to these IDs by class name.
        self._label_ids = {name: i for i, name in enumerate(names)}
        self.spec = cvataa.DetectionFunctionSpec(
            labels=[self._label_spec(name, i) for name, i in self._label_ids.items()]
        )

    @abc.abstractmethod
    def _label_spec(self, name: str, id_: int) -> models.PatchedLabelRequest: ...


class DetrFunctionWithSimpleLabel(DetrFunction):
    LABEL_TYPE: ClassVar[str]

    def _label_spec(self, name: str, id_: int) -> models.PatchedLabelRequest:
        return cvataa.label_spec(name, id_, type=self.LABEL_TYPE)


class DetrFunctionWithShapes(DetrFunction):
    LABEL_TYPE: ClassVar[str]

    def detect(
        self, context: cvataa.DetectionFunctionContext, image: Image.Image
    ) -> list[cvataa.DetectionAnnotation]:
        kwargs = {}
        if context.conf_threshold is not None:
            kwargs["threshold"] = context.conf_threshold

        result = self._model.predict(image.convert("RGB"), include_source_image=False, **kwargs)
        if len(result):
            return list(self._annotations_from_results(context, result, image.size))

        return []

    def _iter_detections(self, context, result) -> Iterator[tuple[int, int]]:
        """Yield detection indices and function-local label IDs for valid predictions."""
        count = len(result)
        names = result.data.get("class_name")
        scores = result.detection_confidence if self.LABEL_TYPE == "skeleton" else result.confidence

        if names is None:
            raise ValueError("Model did not return class names")

        expected_shape = (count,)
        if np.shape(names) != expected_shape:
            raise ValueError("Model returned inconsistent class names")
        if np.shape(scores) != expected_shape:
            raise ValueError("Model returned inconsistent confidence scores")

        self._validate_result(result, count)
        for i, (name, score) in enumerate(zip(names, scores)):
            if name == "__background__" or not np.isfinite(score):
                continue
            if context.conf_threshold is not None and score < context.conf_threshold:
                continue
            if name not in self._label_ids:
                raise ValueError(f"Model returned an unknown class name: {name!r}")
            yield i, self._label_ids[name]

    @abc.abstractmethod
    def _validate_result(self, result, count: int) -> None: ...

    @abc.abstractmethod
    def _annotations_from_results(
        self, context, result, size
    ) -> Iterator[cvataa.DetectionAnnotation]: ...


class DetrDetectionFunction(DetrFunctionWithSimpleLabel, DetrFunctionWithShapes):
    LABEL_TYPE = "rectangle"

    def _validate_result(self, result, count: int) -> None:
        if np.shape(result.xyxy) != (count, 4):
            raise ValueError("Model returned inconsistent bounding boxes")

    def _annotations_from_results(
        self, context, result, size
    ) -> Iterator[cvataa.DetectionAnnotation]:
        for i, label_id in self._iter_detections(context, result):
            box = np.asarray(result.xyxy[i], dtype=float)
            if not np.isfinite(box).all():
                continue
            box = np.clip(box, 0, [size[0], size[1], size[0], size[1]])
            if box[2] > box[0] and box[3] > box[1]:
                yield cvataa.rectangle(label_id, box.tolist())


class DetrSegmentationFunction(DetrFunctionWithSimpleLabel, DetrFunctionWithShapes):
    LABEL_TYPE = "mask"

    def _validate_result(self, result, count: int) -> None:
        if result.mask is None or np.ndim(result.mask) != 3 or len(result.mask) != count:
            raise ValueError("Model returned inconsistent instance masks")

    def _annotations_from_results(
        self, context, result, size
    ) -> Iterator[cvataa.DetectionAnnotation]:
        for i, label_id in self._iter_detections(context, result):
            mask = np.asarray(result.mask[i], dtype=bool)
            if mask.shape != (size[1], size[0]):
                mask = np.asarray(Image.fromarray(mask).resize(size, Image.Resampling.NEAREST))
            if not mask.any():
                continue
            if not context.conv_mask_to_poly:
                yield cvataa.mask(label_id, encode_mask(mask))
                continue

            # Pad and dilate to handle masks touching image boundaries.
            expanded = cv2.copyMakeBorder(
                mask.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
            )
            expanded = cv2.dilate(
                expanded,
                np.ones((2, 2), dtype=np.uint8),
                anchor=(-1, -1),
                iterations=1,
                borderType=cv2.BORDER_CONSTANT,
            )
            contours, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(
                contours, key=lambda contour: cv2.arcLength(contour, closed=False), reverse=True
            )

            epsilon = 1.0
            for contour in contours:
                contour = (contour - 1).astype(np.float32)
                if len(contour) < 3:
                    continue
                approx = cv2.approxPolyDP(contour, epsilon, closed=True)
                if len(approx) < 3:
                    approx = contour
                yield cvataa.polygon(label_id, approx.ravel().tolist())


class DetrPoseEstimationFunction(DetrFunctionWithShapes):
    LABEL_TYPE = "skeleton"

    def __init__(
        self, model: RFDETR, keypoint_names_path: str | None = None, *, pretrained: bool = False
    ) -> None:
        counts = list(model.model_config.num_keypoints_per_class)
        # Legacy checkpoints may reserve slot zero for background.
        if counts and counts[0] == 0:
            counts = counts[1:]

        if pretrained:
            # The default KeypointPreview weights use the COCO person schema.
            names = ["person"]
        else:
            if not model.model.class_names:
                raise ValueError("Custom keypoint checkpoint must provide class names")
            names = list(model.model.class_names)

        if len(counts) != len(names) or any(count <= 0 for count in counts):
            raise ValueError("Model keypoint schema does not match its classes")

        if keypoint_names_path:
            path = Path(keypoint_names_path)
            if path.suffix.lower() == ".json":
                schema = json.loads(path.read_text())
            else:
                shared = [line.strip() for line in path.read_text().splitlines() if line.strip()]
                schema = {name: shared for name in names}
        elif pretrained:
            schema = {"person": list(DEFAULT_KEYPOINT_NAMES)}
        else:
            raise ValueError("Custom keypoint models require keypoint_names_path")

        if not isinstance(schema, dict) or set(schema) != set(names):
            raise ValueError("Keypoint names must define every model class exactly once")

        for name, count in zip(names, counts):
            points = schema[name]
            if (
                not isinstance(points, list)
                or len(points) != count
                or any(not isinstance(p, str) or not p.strip() for p in points)
                or len(set(points)) != count
            ):
                raise ValueError(f"Invalid keypoint names for {name!r}: expected {count} unique names")

        self._keypoint_names = schema
        super().__init__(model, class_names=names)

    def _label_spec(self, name: str, id_: int) -> models.PatchedLabelRequest:
        return cvataa.skeleton_label_spec(
            name,
            id_,
            [cvataa.keypoint_spec(point, i) for i, point in enumerate(self._keypoint_names[name])],
        )

    def _validate_result(self, result, count: int) -> None:
        xy = np.asarray(result.xy)
        if (
            xy.ndim != 3
            or xy.shape[0] != count
            or xy.shape[2] != 2
            or np.shape(result.keypoint_confidence) != xy.shape[:2]
        ):
            raise ValueError("Model returned inconsistent keypoint arrays")
        if xy.shape[1] != max(map(len, self._keypoint_names.values())):
            raise ValueError("Model keypoint output does not match its schema")

    def _annotations_from_results(
        self, context, result, size
    ) -> Iterator[cvataa.DetectionAnnotation]:
        for i, label_id in self._iter_detections(context, result):
            count = len(self._keypoint_names[result.data["class_name"][i]])
            elements = []
            for point_id, (point, score) in enumerate(
                zip(result.xy[i][:count], result.keypoint_confidence[i][:count])
            ):
                valid = np.isfinite(point).all() and np.isfinite(score)
                elements.append(
                    cvataa.keypoint(
                        point_id,
                        np.asarray(point).tolist() if valid else [0.0, 0.0],
                        outside=not valid or float(score) < 0.5,
                    )
                )
            yield cvataa.skeleton(label_id, elements)


FUNCTION_CLASS_BY_TASK = {
    "detect": DetrDetectionFunction,
    "segment": DetrSegmentationFunction,
    "pose": DetrPoseEstimationFunction,
}


def _load_model(model: str | None, *, architecture: str, device: str) -> RFDETR:
    import rfdetr

    if architecture not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model architecture {architecture!r}. "
            f"Choose from {', '.join(SUPPORTED_MODELS)}"
        )
    if model is None:
        return getattr(rfdetr, architecture)(device=device)
    if not Path(model).is_file():
        raise ValueError(f"Checkpoint does not exist: {model}")
    path = str(Path(model).resolve())
    try:
        # Preserve training configuration from self-describing checkpoints.
        loaded = rfdetr.from_checkpoint(path, device=device)
    except KeyError as exc:
        if exc.args != ("args",):
            raise
        # Published checkpoints can lack architecture metadata. The caller
        # must identify their architecture explicitly.
        loaded = getattr(rfdetr, architecture)(pretrain_weights=path, device=device)
    if type(loaded).__name__ != architecture:
        raise ValueError("Checkpoint architecture does not match the selected architecture")
    return loaded


def create(
    model: str | None = None,
    *,
    architecture: str,
    device: str = "cpu",
    keypoint_names_path: str | None = None,
) -> cvataa.DetectionFunction:
    loaded = _load_model(model, architecture=architecture, device=device)
    if loaded.model_config.use_grouppose_keypoints:
        task = "pose"
    elif loaded.model_config.segmentation_head:
        task = "segment"
    else:
        task = "detect"

    kwargs = {}
    if task == "pose":
        kwargs["pretrained"] = model is None and architecture == "RFDETRKeypointPreview"
    if keypoint_names_path is not None:
        if task != "pose":
            raise ValueError("keypoint_names_path is only supported for keypoint models")
        kwargs["keypoint_names_path"] = keypoint_names_path

    return FUNCTION_CLASS_BY_TASK[task](loaded, **kwargs)
