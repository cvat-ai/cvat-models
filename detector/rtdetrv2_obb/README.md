<!--
Copyright (C) CVAT.ai Corporation
SPDX-License-Identifier: MIT
-->

# RT-DETRv2-OBB

This directory provides CVAT auto-annotation functions for the public RT-DETRv2 oriented-box
baseline from [RicePasteM/RiO-DETR][]. The public repository does not contain the complete
paper-specific RiO-DETR architecture. This integration intentionally does not use RiO-DETRv2.

The CVAT adapter and deployment files are distributed under CVAT's MIT license. The separately
cloned upstream source and official [RT-DETR-OBB checkpoints][] remain under the Apache License 2.0.
The prepared upstream checkout contains its license and attribution details. The upstream source is
not copied into this repository.

[RicePasteM/RiO-DETR]: https://github.com/RicePasteM/RiO-DETR
[RT-DETR-OBB checkpoints]: https://huggingface.co/RicePasteM/RT-DETR-OBB

## Supported checkpoints

| Model variant | Sizes | Input size | CVAT wrapper |
| --- | --- | --- | --- |
| DOTA-v1.0 single-scale | N, S, M, L, X | 1024 × 1024 | `dota/func.py` |
| DOTA-v1.0 multi-scale | M, X | 1024 × 1024 | `dota/func.py` |
| DIOR-R | N, S, M, L, X | 800 × 800 | `dior_r/func.py` |

DOTA X single-scale is the default DOTA model. DIOR-R X is the default DIOR-R model. FAIR1M is
not supported because the referenced model zoo does not publish an official FAIR1M checkpoint.

## Preparation

Git is required. Before using either function locally, run the preparation commands from the
directory containing this README.

Clone the upstream repository at the supported revision:

```shell
./clone_upstream.sh
```

The generated `RiO-DETR/` directory is ignored by Git. To use an existing checkout at the exact
supported commit instead, set `RTDETRV2_OBB_UPSTREAM_DIR` before running the script:

```shell
export RTDETRV2_OBB_UPSTREAM_DIR=/absolute/path/to/RiO-DETR
./clone_upstream.sh
```

The integration verifies the checkout revision before loading the model.

Download and verify all 12 official checkpoints:

```shell
python common.py --download-all
```

Download one checkpoint or a whole group with:

```shell
python common.py --group dota-single-scale --model-size X
python common.py --group dior-r
```

Downloads use public `resolve/main` URLs and require no Hugging Face token. Existing files are
reused only after their size and SHA-256 match the official repository metadata. Files are written
to a temporary sibling and atomically installed after verification.

Agent registration performs both preparation steps automatically and reuses the cached source and
checkpoint files.

## Use with CVAT CLI

From the `dota` directory, use the following options for the DOTA function:

```text
--function-file func.py \
    -p checkpoint_group=str:dota-single-scale \
    -p model_size=str:X \
    -p conf_threshold=float:0.4
```

From the `dior_r` directory, use the following options for the DIOR-R function:

```text
--function-file func.py \
    -p checkpoint_group=str:dior-r \
    -p model_size=str:M \
    -p conf_threshold=float:0.4
```

Supported parameters are:

- `checkpoint_group`: model variant to use: `dota-single-scale`, `dota-multi-scale`, or `dior-r`,
  as accepted by the selected dataset wrapper.
- `model_size`: `N`, `S`, `M`, `L`, or `X`; DOTA multi-scale accepts only `M` and `X`.
- `checkpoint`: optional local checkpoint path. The group and model size still select the matching
  architecture configuration.
- `device`: PyTorch device, defaulting to `cpu`; use `cuda` for GPU inference.
- `conf_threshold`: fallback confidence threshold, defaulting to `0.4`. A threshold supplied by a
  CVAT annotation request takes precedence.

Use the same parameters when registering the function and starting its agent. See the
[native-functions guide](https://docs.cvat.ai/docs/annotation/auto-annotation/ai-models/#native-functions)
for dependency installation, authentication, and direct CLI registration. See the
[Docker Compose agent guide](https://docs.cvat.ai/docs/guides/compose-agents-userguide/)
for image building and agent deployment instructions; choose the DOTA or DIOR-R agent directory.

## Output behavior and limitations

The model's `(center_x, center_y, width, height, angle)` output is converted to a CVAT rectangle
with a clockwise image-space rotation in degrees. Partially out-of-frame boxes preserve their true
geometry; only boxes wholly outside the image are discarded.

Inference resizes each complete image to the checkpoint input canvas with preserved aspect ratio
and right/bottom padding. The adapter does not tile large aerial images or merge tile predictions;
applications requiring the original DOTA patch workflow should split images before inference.
