# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""Shared RT-DETRv2-OBB loading, inference, and checkpoint management."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cvat_sdk.auto_annotation as cvataa
import PIL.Image
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

if __package__:
    from .upstream_loader import get_upstream_dir, load_upstream_runtime
else:
    from upstream_loader import get_upstream_dir, load_upstream_runtime

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
HF_BASE_URL = "https://huggingface.co/RicePasteM/RT-DETR-OBB/resolve/main"

GROUP_DOTA_SINGLE_SCALE = "dota-single-scale"
GROUP_DOTA_MULTI_SCALE = "dota-multi-scale"
GROUP_DIOR_R = "dior-r"

DEFAULT_CONF_THRESHOLD = 0.4
DEFAULT_MODEL_SIZE = "X"


@dataclass(frozen=True)
class CheckpointSpec:
    group: str
    model_size: str
    remote_directory: str
    filename: str
    byte_size: int
    sha256: str
    config_filename: str
    input_size: int

    @property
    def url(self) -> str:
        return f"{HF_BASE_URL}/{self.remote_directory}/{self.filename}"

    def path(self, checkpoint_root: Path = CHECKPOINTS_DIR) -> Path:
        return checkpoint_root / self.group / self.filename

    @property
    def model_name(self) -> str:
        return f"RT-DETRv2-OBB-{self.model_size}"

    def config_path(self, upstream_root: Path) -> Path:
        return upstream_root / "configs" / "rtdetrv2_obb" / self.config_filename


def _spec(
    group: str,
    size: str,
    remote_directory: str,
    filename: str,
    byte_size: int,
    sha256: str,
    config_filename: str,
    input_size: int,
) -> CheckpointSpec:
    return CheckpointSpec(
        group=group,
        model_size=size,
        remote_directory=remote_directory,
        filename=filename,
        byte_size=byte_size,
        sha256=sha256,
        config_filename=config_filename,
        input_size=input_size,
    )


CHECKPOINTS: dict[tuple[str, str], CheckpointSpec] = {
    (GROUP_DIOR_R, "N"): _spec(
        GROUP_DIOR_R,
        "N",
        "diorr",
        "rtdetrv2_obb_hgnetv2_n_diorr.pth",
        16_227_426,
        "6d00665413e48d60bc942defb7c12204b3d080cfd08b9a7b407dc4d80e80f0fc",
        "rtdetrv2_obb_hgnetv2_n_diorr.yml",
        800,
    ),
    (GROUP_DIOR_R, "S"): _spec(
        GROUP_DIOR_R,
        "S",
        "diorr",
        "rtdetrv2_obb_hgnetv2_s_diorr.pth",
        33_172_282,
        "b37c73b8030c1ef784e8f35ca9fab3285e3e60ba6c6035057ea38257ffee01f1",
        "rtdetrv2_obb_hgnetv2_s_diorr.yml",
        800,
    ),
    (GROUP_DIOR_R, "M"): _spec(
        GROUP_DIOR_R,
        "M",
        "diorr",
        "rtdetrv2_obb_hgnetv2_m_diorr.pth",
        77_014_180,
        "2a6384fe713679e4c3269d5176e8549c773fdfa87bb818da782e79b2b372d362",
        "rtdetrv2_obb_hgnetv2_m_diorr.yml",
        800,
    ),
    (GROUP_DIOR_R, "L"): _spec(
        GROUP_DIOR_R,
        "L",
        "diorr",
        "rtdetrv2_obb_hgnetv2_l_diorr.pth",
        112_683_971,
        "3a403d589585403091a6b154a32499d7ccf6e3196c7ac8b91e9bf1c182457125",
        "rtdetrv2_obb_hgnetv2_l_diorr.yml",
        800,
    ),
    (GROUP_DIOR_R, "X"): _spec(
        GROUP_DIOR_R,
        "X",
        "diorr",
        "rtdetrv2_obb_hgnetv2_x_diorr.pth",
        255_542_330,
        "67ef7436857489460b6abd9cce984c723dfe4f183f62ad0e43ba49136b2711e8",
        "rtdetrv2_obb_hgnetv2_x_diorr.yml",
        800,
    ),
    (GROUP_DOTA_SINGLE_SCALE, "N"): _spec(
        GROUP_DOTA_SINGLE_SCALE,
        "N",
        "dota_1_ss",
        "rtdetrv2_obb_hgnetv2_n_dota_1_ss.pth",
        16_251_702,
        "b87de5d18f7abfa0ea83aaa27e5f7a4d6545553c6ee559afbff9ac95a7d41006",
        "rtdetrv2_obb_hgnetv2_n_dota_1_ss.yml",
        1024,
    ),
    (GROUP_DOTA_SINGLE_SCALE, "S"): _spec(
        GROUP_DOTA_SINGLE_SCALE,
        "S",
        "dota_1_ss",
        "rtdetrv2_obb_hgnetv2_s_dota_1_ss.pth",
        33_289_786,
        "ef0c728aaeb4d85950134431617fb576b3ad6a9ac85878088ce97c0075df2026",
        "rtdetrv2_obb_hgnetv2_s_dota_1_ss.yml",
        1024,
    ),
    (GROUP_DOTA_SINGLE_SCALE, "M"): _spec(
        GROUP_DOTA_SINGLE_SCALE,
        "M",
        "dota_1_ss",
        "rtdetrv2_obb_hgnetv2_m_dota_1_ss.pth",
        77_127_788,
        "2cbfc60909a7d1209192f35881aed407f8790317fe9ce051cefa3e17ed88e0c6",
        "rtdetrv2_obb_hgnetv2_m_dota_1_ss.yml",
        1024,
    ),
    (GROUP_DOTA_SINGLE_SCALE, "L"): _spec(
        GROUP_DOTA_SINGLE_SCALE,
        "L",
        "dota_1_ss",
        "rtdetrv2_obb_hgnetv2_l_dota_1_ss.pth",
        112_792_183,
        "503b790952f1d8b3ad9773b5a61a175246b8ba81aae50dbf02f1fc3c5b792cdd",
        "rtdetrv2_obb_hgnetv2_l_dota_1_ss.yml",
        1024,
    ),
    (GROUP_DOTA_SINGLE_SCALE, "X"): _spec(
        GROUP_DOTA_SINGLE_SCALE,
        "X",
        "dota_1_ss",
        "rtdetrv2_obb_hgnetv2_x_dota_1_ss.pth",
        255_633_982,
        "9121e1da4c79e37fa204d8caaad3107e6edc4b8780d9d936f0747fd35330476d",
        "rtdetrv2_obb_hgnetv2_x_dota_1_ss.yml",
        1024,
    ),
    (GROUP_DOTA_MULTI_SCALE, "M"): _spec(
        GROUP_DOTA_MULTI_SCALE,
        "M",
        "dota_1_ms",
        "rtdetrv2_obb_hgnetv2_m_dota_1_ms.pth",
        77_127_788,
        "a01e166b85ad6b7a19390c26c6a1773350ae483c1d7ec7379f9658752b6d5b21",
        "rtdetrv2_obb_hgnetv2_m_dota_1_ms.yml",
        1024,
    ),
    (GROUP_DOTA_MULTI_SCALE, "X"): _spec(
        GROUP_DOTA_MULTI_SCALE,
        "X",
        "dota_1_ms",
        "rtdetrv2_obb_hgnetv2_x_dota_1_ms.pth",
        255_633_982,
        "034c52993da3cfb8d3643a44e69ea56d1907cdd6c8e7907590def041a5c57008",
        "rtdetrv2_obb_hgnetv2_x_dota_1_ms.yml",
        1024,
    ),
}


def get_checkpoint_spec(group: str, model_size: str) -> CheckpointSpec:
    normalized_size = model_size.upper()
    try:
        return CHECKPOINTS[(group, normalized_size)]
    except KeyError as ex:
        supported = ", ".join(
            spec.model_size for spec in CHECKPOINTS.values() if spec.group == group
        )
        if not supported:
            groups = ", ".join(sorted({spec.group for spec in CHECKPOINTS.values()}))
            raise ValueError(f"Unsupported checkpoint group {group!r}; choose from {groups}") from ex
        raise ValueError(
            f"Unsupported model size {model_size!r} for {group}; choose from {supported}"
        ) from ex


def _file_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def checkpoint_is_valid(path: Path, spec: CheckpointSpec) -> bool:
    if not path.is_file() or path.stat().st_size != spec.byte_size:
        return False
    return _file_digest(path) == (spec.byte_size, spec.sha256)


def download_checkpoint(
    spec: CheckpointSpec,
    *,
    checkpoint_root: Path = CHECKPOINTS_DIR,
    opener=urllib.request.urlopen,
) -> Path:
    destination = spec.path(checkpoint_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_is_valid(destination, spec):
        print(f"Using verified checkpoint: {destination}")
        return destination

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            digest = hashlib.sha256()
            byte_count = 0
            print(f"Downloading {spec.model_name} ({spec.group}) from {spec.url}")
            with opener(spec.url) as response:
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())

        if byte_count != spec.byte_size:
            raise RuntimeError(
                f"Downloaded size mismatch for {spec.filename}: expected {spec.byte_size}, got {byte_count}"
            )
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != spec.sha256:
            raise RuntimeError(
                f"Downloaded SHA-256 mismatch for {spec.filename}: "
                f"expected {spec.sha256}, got {actual_sha256}"
            )

        os.replace(temporary_path, destination)
        temporary_path = None
        print(f"Saved verified checkpoint: {destination}")
        return destination
    except (OSError, urllib.error.URLError) as ex:
        raise RuntimeError(f"Failed to download {spec.url}: {ex}") from ex
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def download_checkpoints(
    *,
    group: str | None = None,
    model_size: str | None = None,
    checkpoint_root: Path = CHECKPOINTS_DIR,
) -> list[Path]:
    if model_size is not None and group is None:
        raise ValueError("model_size requires group")
    if group is not None and group not in {key[0] for key in CHECKPOINTS}:
        get_checkpoint_spec(group, model_size or DEFAULT_MODEL_SIZE)

    selected = [
        spec
        for (spec_group, spec_size), spec in CHECKPOINTS.items()
        if (group is None or spec_group == group)
        and (model_size is None or spec_size == model_size.upper())
    ]
    if not selected:
        get_checkpoint_spec(group or "", model_size or "")
    return [download_checkpoint(spec, checkpoint_root=checkpoint_root) for spec in selected]


def preprocess_image(image: PIL.Image.Image, input_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")

    scale = min(input_size / width, input_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = TF.resize(
        image,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    padded = TF.pad(
        resized,
        [0, 0, input_size - resized_width, input_size - resized_height],
        fill=0,
    )
    return TF.to_tensor(padded).unsqueeze(0), (height, width)


def _clip_polygon_to_image(
    polygon: Sequence[tuple[float, float]], width: int, height: int
) -> list[tuple[float, float]]:
    def clip(
        points: list[tuple[float, float]],
        inside,
        intersection,
    ) -> list[tuple[float, float]]:
        if not points:
            return []
        output: list[tuple[float, float]] = []
        previous = points[-1]
        previous_inside = inside(previous)
        for current in points:
            current_inside = inside(current)
            if current_inside != previous_inside:
                output.append(intersection(previous, current))
            if current_inside:
                output.append(current)
            previous = current
            previous_inside = current_inside
        return output

    def vertical_intersection(a, b, x):
        if b[0] == a[0]:
            return (x, a[1])
        ratio = (x - a[0]) / (b[0] - a[0])
        return (x, a[1] + ratio * (b[1] - a[1]))

    def horizontal_intersection(a, b, y):
        if b[1] == a[1]:
            return (a[0], y)
        ratio = (y - a[1]) / (b[1] - a[1])
        return (a[0] + ratio * (b[0] - a[0]), y)

    clipped = list(polygon)
    clipped = clip(clipped, lambda p: p[0] >= 0, lambda a, b: vertical_intersection(a, b, 0))
    clipped = clip(
        clipped, lambda p: p[0] <= width, lambda a, b: vertical_intersection(a, b, width)
    )
    clipped = clip(clipped, lambda p: p[1] >= 0, lambda a, b: horizontal_intersection(a, b, 0))
    return clip(
        clipped, lambda p: p[1] <= height, lambda a, b: horizontal_intersection(a, b, height)
    )


def rotated_box_intersects_image(
    cx: float, cy: float, width: float, height: float, angle_radians: float, image_size: tuple[int, int]
) -> bool:
    image_width, image_height = image_size
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    corners = []
    for dx, dy in (
        (-width / 2, -height / 2),
        (width / 2, -height / 2),
        (width / 2, height / 2),
        (-width / 2, height / 2),
    ):
        corners.append((cx + dx * cosine - dy * sine, cy + dx * sine + dy * cosine))
    return bool(_clip_polygon_to_image(corners, image_width, image_height))


class RTDETRv2OBBFunction:
    def __init__(
        self,
        *,
        class_names: Sequence[str],
        checkpoint_group: str,
        model_size: str = DEFAULT_MODEL_SIZE,
        checkpoint: str | None = None,
        device: str = "cpu",
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        checkpoint_root: Path = CHECKPOINTS_DIR,
    ) -> None:
        if not class_names or len(set(class_names)) != len(class_names):
            raise ValueError("Class names must be nonempty and unique")
        if not 0 <= conf_threshold <= 1:
            raise ValueError("conf_threshold must be between 0 and 1")

        self._class_names = tuple(class_names)
        self._spec = get_checkpoint_spec(checkpoint_group, model_size)
        self._device = torch.device(device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")
        self._conf_threshold = float(conf_threshold)

        if checkpoint is None:
            checkpoint_path = download_checkpoint(self._spec, checkpoint_root=checkpoint_root)
        else:
            checkpoint_path = Path(checkpoint).expanduser().resolve()
            if not checkpoint_path.is_file():
                raise ValueError(f"Checkpoint does not exist: {checkpoint_path}")

        self._model, self._postprocessor = load_model(
            self._spec, checkpoint_path, self._device, expected_classes=len(self._class_names)
        )
        self.spec = cvataa.DetectionFunctionSpec(
            labels=[cvataa.label_spec(name, label_id, type="rectangle") for label_id, name in enumerate(class_names)]
        )

    def detect(
        self, context: cvataa.DetectionFunctionContext, image: PIL.Image.Image
    ) -> list[cvataa.DetectionAnnotation]:
        tensor, (height, width) = preprocess_image(image, self._spec.input_size)
        tensor = tensor.to(self._device)
        original_sizes = torch.tensor([[height, width]], device=self._device)
        with torch.inference_mode():
            outputs = self._postprocessor(self._model(tensor), original_sizes)
        labels, boxes, scores = (value[0].detach().cpu() for value in outputs)
        threshold = (
            self._conf_threshold if context.conf_threshold is None else float(context.conf_threshold)
        )
        return list(
            annotations_from_output(
                labels,
                boxes,
                scores,
                image_size=(width, height),
                class_count=len(self._class_names),
                conf_threshold=threshold,
            )
        )


def load_model(
    spec: CheckpointSpec,
    checkpoint_path: Path,
    device: torch.device,
    *,
    expected_classes: int,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    upstream = load_upstream_runtime(get_upstream_dir())
    config = upstream.load_config(str(spec.config_path(upstream.root)), {})
    if int(config["num_classes"]) != expected_classes:
        raise ValueError(
            f"Configuration has {config['num_classes']} classes, expected {expected_classes}"
        )
    config["HGNetv2"]["pretrained"] = False
    global_config = upstream.merge_config(config, inplace=False)
    model = upstream.create(config["model"], global_config)
    postprocessor = upstream.create(config["postprocessor"], global_config)

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as ex:
        raise RuntimeError(f"Unable to load checkpoint {checkpoint_path}: {ex}") from ex
    if "ema" in checkpoint:
        state = checkpoint["ema"].get("module", checkpoint["ema"])
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        raise ValueError(f"Checkpoint {checkpoint_path} has neither 'ema' nor 'model' weights")
    model.load_state_dict(state, strict=True)
    model = model.deploy().to(device)
    postprocessor = postprocessor.deploy().to(device)
    return model, postprocessor


def annotations_from_output(
    labels: Iterable,
    boxes: Iterable,
    scores: Iterable,
    *,
    image_size: tuple[int, int],
    class_count: int,
    conf_threshold: float,
) -> Iterable[cvataa.DetectionAnnotation]:
    for raw_label, raw_box, raw_score in zip(labels, boxes, scores):
        label_id = int(raw_label.item() if hasattr(raw_label, "item") else raw_label)
        score = float(raw_score.item() if hasattr(raw_score, "item") else raw_score)
        box = raw_box.tolist() if hasattr(raw_box, "tolist") else list(raw_box)
        if len(box) != 5 or not math.isfinite(score) or score < conf_threshold:
            continue
        if label_id < 0 or label_id >= class_count:
            continue
        cx, cy, width, height, angle_radians = map(float, box)
        if not all(map(math.isfinite, (cx, cy, width, height, angle_radians))):
            continue
        if width <= 0 or height <= 0:
            continue
        if not rotated_box_intersects_image(
            cx, cy, width, height, angle_radians, image_size
        ):
            continue
        angle_degrees = math.degrees(angle_radians) % 180.0
        yield cvataa.rectangle(
            label_id,
            [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
            rotation=angle_degrees,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official RT-DETRv2-OBB checkpoints")
    parser.add_argument("--download-all", action="store_true", help="download all 12 checkpoints")
    parser.add_argument("--group", choices=sorted({key[0] for key in CHECKPOINTS}))
    parser.add_argument("--model-size", choices=("N", "S", "M", "L", "X"))
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINTS_DIR)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.download_all and args.group is None:
        raise SystemExit("Specify --download-all or --group")
    group = None if args.download_all else args.group
    model_size = None if args.download_all else args.model_size
    paths = download_checkpoints(
        group=group, model_size=model_size, checkpoint_root=args.checkpoint_root
    )
    print(f"Verified {len(paths)} checkpoint(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
