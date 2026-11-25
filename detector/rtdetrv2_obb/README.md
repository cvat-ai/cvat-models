<!--
Copyright (C) CVAT.ai Corporation
SPDX-License-Identifier: MIT
-->

# RT-DETRv2-OBB

This directory provides CVAT auto-annotation functions for the public RT-DETRv2 oriented-box
baseline from [RicePasteM/RiO-DETR](https://github.com/RicePasteM/RiO-DETR).

The CVAT adapter and deployment files are distributed under CVAT's MIT license. The separately
cloned upstream source and official
[RT-DETR-OBB checkpoints](https://huggingface.co/RicePasteM/RT-DETR-OBB) remain under the Apache
License 2.0. The upstream source is not copied into this repository.

## Supported checkpoints

| Model variant | Sizes | Input size |
| --- | --- | --- |
| DOTA-v1.0 single-scale | N, S, M, L, X | 1024 × 1024 |
| DOTA-v1.0 multi-scale | M, X | 1024 × 1024 |
| DIOR-R | N, S, M, L, X | 800 × 800 |

DOTA X single-scale is the default DOTA model. DIOR-R X is the default DIOR-R model.

## Local preparation

To use the function directly, clone the upstream repository at the target revision:

```shell
./clone_upstream.sh
```

Download and verify all 12 official checkpoints:

```shell
python common.py --download-all
```

Download one checkpoint or a whole group with:

```shell
python common.py --group dota-single-scale --model-size X
python common.py --group dior-r
```

Downloads use public URLs and require no Hugging Face token.

The agent image includes the pinned upstream source. Agent registration downloads the selected
checkpoint and reuses it from the checkpoint cache.

## Use with CVAT CLI

From the directory containing this README, use the following options for DOTA:

```text
--function-file func.py \
    -p dataset=str:dota \
    -p model_size=str:X \
    -p conf_threshold=float:0.4
```

For DIOR-R, change the required dataset parameter:

```text
--function-file func.py \
    -p dataset=str:dior-r \
    -p model_size=str:M \
    -p conf_threshold=float:0.4
```

Supported parameters are:

- `dataset`: required dataset and class mapping: `dota` or `dior-r`.
- `checkpoint_group`: optional model variant. DOTA accepts `dota-single-scale` (the default) or
  `dota-multi-scale`; DIOR-R uses `dior-r`.
- `model_size`: `N`, `S`, `M`, `L`, or `X`; DOTA multi-scale accepts only `M` and `X`.
- `checkpoint`: optional local checkpoint path. The group and model size still select the matching
  architecture configuration.
- `device`: PyTorch device, defaulting to `cpu`; use `cuda` for GPU inference.
- `conf_threshold`: fallback confidence threshold, defaulting to `0.4`. A threshold supplied by a
  CVAT annotation request takes precedence.

Use these parameters when registering the function. See the
[native-functions guide](https://docs.cvat.ai/docs/annotation/auto-annotation/ai-models/#native-functions)
for dependency installation, authentication, and direct CLI registration. See the
[Docker Compose agent guide](https://docs.cvat.ai/docs/guides/compose-agents-userguide/)
for image building and agent deployment instructions. The `agent` image supports both
datasets; include `-p dataset=str:dota` or `-p dataset=str:dior-r` in `MODEL_CONFIG_PARAMS`.

## Output behavior and limitations

The model's `(center_x, center_y, width, height, angle)` output is converted to a CVAT rectangle
with a clockwise image-space rotation in degrees. Partially out-of-frame boxes preserve their true
geometry; only boxes wholly outside the image are discarded.

Inference resizes each complete image to the checkpoint input canvas with preserved aspect ratio
and right/bottom padding. The adapter does not tile large aerial images or merge tile predictions;
applications requiring the original DOTA patch workflow should split images before inference.
