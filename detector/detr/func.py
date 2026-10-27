# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import cvat_sdk.auto_annotation as cvataa
import numpy as np
from cvat_sdk.masks import encode_mask
from PIL import Image
from skimage import measure

if TYPE_CHECKING:
    from rfdetr import RFDETR

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
COCO_KEYPOINT_NAMES = (
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


class DetrFunction:
    LABEL_TYPE: ClassVar[str] = "rectangle"

    def __init__(self, model: "RFDETR", *, class_names: list[str] | None = None) -> None:
        self._model = model
        names = list(model.class_names) if class_names is None else class_names
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("RF-DETR must provide nonempty class names")
        if len(set(names)) != len(names):
            raise ValueError("RF-DETR class names must be unique")
        self._label_ids = {name: i for i, name in enumerate(names)}
        self.spec = cvataa.DetectionFunctionSpec(
            labels=[self._label_spec(name, i) for name, i in self._label_ids.items()]
        )

    def _label_spec(self, name, id_):
        return cvataa.label_spec(name, id_, type=self.LABEL_TYPE)

    def detect(
        self, context: cvataa.DetectionFunctionContext, image: Image.Image
    ) -> list[cvataa.DetectionAnnotation]:
        threshold = 0.5 if context.conf_threshold is None else context.conf_threshold
        result = self._model.predict(
            image.convert("RGB"), threshold=threshold, include_source_image=False
        )
        count = len(result)
        if not count:
            return []
        names = result.data.get("class_name")
        scores = result.detection_confidence if self.LABEL_TYPE == "skeleton" else result.confidence
        if names is None or np.shape(names) != (count,) or np.shape(scores) != (count,):
            raise ValueError("RF-DETR returned inconsistent class names or confidence scores")
        self._validate_result(result, count)
        shapes = []
        for i, (name, score) in enumerate(zip(names, scores)):
            if name == "__background__" or not np.isfinite(score) or score < threshold:
                continue
            if name not in self._label_ids:
                raise ValueError(f"RF-DETR returned an unknown class name: {name!r}")
            shapes.extend(self._shapes(context, result, i, self._label_ids[name], image.size))
        return shapes

    def _validate_result(self, result, count):
        if np.shape(result.xyxy) != (count, 4):
            raise ValueError("RF-DETR returned inconsistent bounding boxes")

    def _shapes(self, context, result, i, label_id, size) -> Iterator[cvataa.DetectionAnnotation]:
        box = np.asarray(result.xyxy[i], dtype=float)
        if not np.isfinite(box).all():
            return
        box = np.clip(box, 0, [size[0], size[1], size[0], size[1]])
        if box[2] > box[0] and box[3] > box[1]:
            yield cvataa.rectangle(label_id, box.tolist())


class DetrSegmentationFunction(DetrFunction):
    LABEL_TYPE: ClassVar[str] = "mask"

    def _validate_result(self, result, count):
        if result.mask is None or np.ndim(result.mask) != 3 or len(result.mask) != count:
            raise ValueError("RF-DETR returned inconsistent instance masks")

    def _shapes(self, context, result, i, label_id, size) -> Iterator[cvataa.DetectionAnnotation]:
        mask = np.asarray(result.mask[i], dtype=bool)
        if mask.shape != (size[1], size[0]):
            mask = np.asarray(Image.fromarray(mask).resize(size, Image.Resampling.NEAREST))
        if not mask.any():
            return
        if not context.conv_mask_to_poly:
            yield cvataa.mask(label_id, encode_mask(mask))
            return
        # Padding closes contours for instances touching the image boundary.
        for contour in measure.find_contours(np.pad(mask, 1), level=0.5):
            ys, xs = contour.T
            if len(contour) < 3 or np.sum(xs * np.roll(ys, -1)) < np.sum(ys * np.roll(xs, -1)):
                continue  # Interior holes cannot be represented by CVAT polygons.
            approx = measure.approximate_polygon(contour, tolerance=2.5)
            if len(approx) < 4:
                approx = contour
            points = np.clip((approx - 1)[:, ::-1], 0, [size[0] - 1, size[1] - 1])
            if len(np.unique(points, axis=0)) >= 3:
                yield cvataa.polygon(label_id, points.ravel().tolist())


class DetrKeypointFunction(DetrFunction):
    LABEL_TYPE: ClassVar[str] = "skeleton"

    def __init__(self, model: "RFDETR", keypoint_names_path: str | None = None) -> None:
        counts = list(model.model_config.num_keypoints_per_class)
        # Legacy checkpoints may reserve slot zero for background.
        if counts and counts[0] == 0:
            counts = counts[1:]
        names = list(model.class_names)
        # The published preview checkpoint has no class names; the library falls back
        # to all 80 COCO names even though only the person class has keypoints.
        if model.model.class_names is None and counts == [17] and names and names[0] == "person":
            names = ["person"]
        if len(counts) != len(names) or any(count <= 0 for count in counts):
            raise ValueError("RF-DETR keypoint schema does not match its classes")
        if keypoint_names_path:
            path = Path(keypoint_names_path)
            if path.suffix.lower() == ".json":
                schema = json.loads(path.read_text())
            else:
                shared = [line.strip() for line in path.read_text().splitlines() if line.strip()]
                schema = {name: shared for name in names}
        elif names == ["person"] and counts == [17]:
            schema = {"person": list(COCO_KEYPOINT_NAMES)}
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
                raise ValueError(
                    f"Invalid keypoint names for {name!r}: expected {count} unique names"
                )
        self._keypoint_names = schema
        super().__init__(model, class_names=names)

    def _label_spec(self, name, id_):
        return cvataa.skeleton_label_spec(
            name,
            id_,
            [cvataa.keypoint_spec(point, i) for i, point in enumerate(self._keypoint_names[name])],
        )

    def _validate_result(self, result, count):
        xy = np.asarray(result.xy)
        if (
            xy.ndim != 3
            or xy.shape[0] != count
            or xy.shape[2] != 2
            or np.shape(result.keypoint_confidence) != xy.shape[:2]
        ):
            raise ValueError("RF-DETR returned inconsistent keypoint arrays")
        if xy.shape[1] != max(map(len, self._keypoint_names.values())):
            raise ValueError("RF-DETR keypoint output does not match its schema")

    def _shapes(self, context, result, i, label_id, size) -> Iterator[cvataa.DetectionAnnotation]:
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


def create(
    model: str,
    *,
    device: str = "cpu",
    pretrain_weights: str | None = None,
    keypoint_names_path: str | None = None,
) -> cvataa.DetectionFunction:
    import rfdetr

    if model in SUPPORTED_MODELS:
        kwargs = {"device": device}
        if pretrain_weights is not None:
            if not Path(pretrain_weights).is_file():
                raise ValueError(f"Checkpoint does not exist: {pretrain_weights}")
            path = str(Path(pretrain_weights).resolve())
            try:
                loaded = rfdetr.from_checkpoint(path, device=device)
            except KeyError as exc:
                if exc.args != ("args",):
                    raise
                # Some published weight files contain only model tensors. Their
                # architecture is supplied explicitly by the selected class.
                loaded = getattr(rfdetr, model)(pretrain_weights=path, device=device)
            if type(loaded).__name__ != model:
                raise ValueError("Checkpoint architecture does not match the selected model")
        else:
            loaded = getattr(rfdetr, model)(**kwargs)
    elif Path(model).is_file():
        if pretrain_weights is not None:
            raise ValueError("pretrain_weights requires a named model, not a checkpoint path")
        try:
            loaded = rfdetr.from_checkpoint(str(Path(model).resolve()), device=device)
        except KeyError as exc:
            raise ValueError(
                "Checkpoint lacks architecture metadata; select a supported model name "
                "and pass the file as pretrain_weights"
            ) from exc
    else:
        raise ValueError(
            f"Unsupported RF-DETR model {model!r}. Choose from {', '.join(SUPPORTED_MODELS)} "
            "or provide an existing checkpoint path"
        )
    if type(loaded).__name__ not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported RF-DETR checkpoint architecture: {type(loaded).__name__}")
    if loaded.model_config.use_grouppose_keypoints:
        return DetrKeypointFunction(loaded, keypoint_names_path)
    if keypoint_names_path is not None:
        raise ValueError("keypoint_names_path is only supported for keypoint models")
    if loaded.model_config.segmentation_head:
        return DetrSegmentationFunction(loaded)
    return DetrFunction(loaded)
