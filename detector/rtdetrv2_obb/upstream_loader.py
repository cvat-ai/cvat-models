# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""Load the pinned upstream inference modules without importing training code."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UPSTREAM_COMMIT = "125c132d41946378e4f0e6067f0f8d28ade008fc"
UPSTREAM_DIR_ENV = "RTDETRV2_OBB_UPSTREAM_DIR"


def get_upstream_dir() -> Path:
    configured = os.environ.get(UPSTREAM_DIR_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent / "RiO-DETR"


def verify_upstream_checkout(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not (path / "engine" / "core" / "workspace.py").is_file():
        raise RuntimeError(
            f"RiO-DETR is not cloned at {path}. Follow the git clone step in detector/rtdetrv2_obb/README.md."
        )
    try:
        revision = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as ex:
        raise RuntimeError(f"Unable to verify the RiO-DETR checkout at {path}") from ex
    if revision != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"RiO-DETR checkout at {path} is revision {revision}, expected {UPSTREAM_COMMIT}"
        )
    return path


@dataclass(frozen=True)
class UpstreamRuntime:
    root: Path
    create: Any
    load_config: Any
    merge_config: Any


_RUNTIMES: dict[Path, UpstreamRuntime] = {}


def _package(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(path / "__init__.py")
    module.__package__ = name
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _source_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load upstream module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def load_upstream_runtime(root: Path) -> UpstreamRuntime:
    root = verify_upstream_checkout(root)
    if runtime := _RUNTIMES.get(root):
        return runtime

    namespace = f"_cvat_rtdetrv2_obb_{abs(hash(root))}"
    engine_root = root / "engine"
    _package(namespace, engine_root)

    core_package = _package(f"{namespace}.core", engine_root / "core")
    workspace = _source_module(
        f"{namespace}.core.workspace", engine_root / "core" / "workspace.py"
    )
    core_package.GLOBAL_CONFIG = workspace.GLOBAL_CONFIG
    core_package.create = workspace.create
    core_package.register = workspace.register
    yaml_utils = _source_module(
        f"{namespace}.core.yaml_utils", engine_root / "core" / "yaml_utils.py"
    )
    core_package.load_config = yaml_utils.load_config
    core_package.merge_config = yaml_utils.merge_config
    core_package.merge_dict = yaml_utils.merge_dict

    _package(f"{namespace}.backbone", engine_root / "backbone")
    _source_module(
        f"{namespace}.backbone.common", engine_root / "backbone" / "common.py"
    )
    _source_module(
        f"{namespace}.backbone.hgnetv2", engine_root / "backbone" / "hgnetv2.py"
    )

    _package(f"{namespace}.rtv4", engine_root / "rtv4")
    for module_name in (
        "utils",
        "denoising_obb",
        "rtv4",
        "hybrid_encoder",
        "rtdetrv2_obb_decoder",
        "postprocessor",
    ):
        _source_module(
            f"{namespace}.rtv4.{module_name}", engine_root / "rtv4" / f"{module_name}.py"
        )

    runtime = UpstreamRuntime(
        root=root,
        create=workspace.create,
        load_config=yaml_utils.load_config,
        merge_config=yaml_utils.merge_config,
    )
    _RUNTIMES[root] = runtime
    return runtime
