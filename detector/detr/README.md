# RF-DETR

This CVAT auto-annotation function runs RF-DETR models locally through the CVAT CLI.
It supports object detection, instance segmentation, and keypoint detection with
`rfdetr==1.9.2`. The wrapper is MIT-licensed; the supported upstream models are Apache-2.0.
RF-DETR Plus and deprecated architectures are not supported, including in local checkpoints.

## Supported models

| Task | `model` values | Output |
| --- | --- | --- |
| Detection | `RFDETRNano`, `RFDETRSmall`, `RFDETRMedium`, `RFDETRLarge` | Rectangles |
| Instance segmentation | `RFDETRSegNano`, `RFDETRSegSmall`, `RFDETRSegMedium`, `RFDETRSegLarge`, `RFDETRSegXLarge`, `RFDETRSeg2XLarge` | Masks or polygons |
| Keypoint detection | `RFDETRKeypointPreview` | Skeletons |

KeypointPreview is a preview model. Deprecated `RFDETRBase`, `RFDETRLargeDeprecated`, and
`RFDETRSegPreview` are excluded. This function uses the RF-DETR library, separately from
the Hugging Face DETR models supported by `detector/transformers`.

## Run with CVAT CLI

From the repository root, in a Python environment supported by RF-DETR (Python 3.12 is recommended):

```bash
python -m pip install cvat-cli -r detector/detr/requirements.txt
export CVAT_SERVER_URL="https://app.cvat.ai"
export CVAT_ACCESS_TOKEN="replace-with-your-token"
export CVAT_ORG=""  # Or your organization slug.

cvat-cli --server-host "$CVAT_SERVER_URL" --org "$CVAT_ORG" function create-native "RF-DETR" \
  --function-file detector/detr/func.py \
  -p model=str:RFDETRLarge -p device=str:cpu

cvat-cli --server-host "$CVAT_SERVER_URL" --org "$CVAT_ORG" function run-agent <function_id> \
  --function-file detector/detr/func.py \
  -p model=str:RFDETRLarge -p device=str:cpu
```

Replace `<function_id>` with the registered function's numeric ID. Keep the agent running.
In CVAT, select **AI assistance → Auto-labeling → Model → RF-DETR**, map its labels to your task,
and run it. Task automatic annotation is also supported. Pose models require matching skeleton labels.

Use identical model parameters and workspace settings for registration and execution. Register
different models under distinct function names. For segmentation, substitute `RFDETRSegLarge`;
for human poses, substitute `RFDETRKeypointPreview`. Segmentation returns masks unless the request
enables mask-to-polygon conversion. Polygon conversion emits exterior contours and omits holes.

For GPU inference, install CUDA-compatible PyTorch dependencies and replace `device=str:cpu`
with `device=str:cuda` or `device=str:cuda:0` in both commands. CPU is the default device.
The CVAT confidence threshold is forwarded to RF-DETR; when absent it defaults to `0.5`.
Keypoints below confidence `0.5` are marked outside.

Pretrained weights download automatically and use RF-DETR's cache, normally `~/.roboflow/models`.
Set `RF_HOME` to change that directory. No hosted inference service or Roboflow API token is required.

## Local checkpoints

Use a checkpoint whose architecture and class metadata can be inferred by RF-DETR:

```bash
--function-file detector/detr/func.py -p model=str:/absolute/path/to/checkpoint.pth -p device=str:cpu
```

Alternatively, explicitly select its architecture:

```bash
--function-file detector/detr/func.py \
  -p model=str:RFDETRLarge -p pretrain_weights=str:/absolute/path/to/checkpoint.pth
```

The architecture in the checkpoint must match the selected model. For published weight files
without architecture metadata, use an explicit model name with `pretrain_weights`.
`pretrain_weights` accepts an
existing local file and cannot be combined with a checkpoint path in `model`. The upstream safe
checkpoint loader is used; unrestricted pickle loading is not enabled. Class names must be unique
and present in the model metadata. Custom checkpoints must use one of the supported architectures.

## Custom keypoint names

The pretrained human-pose model uses the 17 COCO keypoints, in the same order as YOLO.
For custom schemas, supply `-p keypoint_names_path=str:/absolute/path/to/names.txt` in both CLI commands.
A text file contains one name per line; blank lines and surrounding whitespace are ignored.
The list applies to every class, so each class must have the same keypoint count.

For different schemas per class, use a `.json` file mapping every class name to an ordered list:

```json
{
  "cat": ["nose", "tail"],
  "car": ["front_left", "front_right", "rear_left", "rear_right"]
}
```

Names must be nonempty and unique within each class, and counts must match the checkpoint.
This parameter is only valid for keypoint models. Classes without keypoints in a mixed
detection/keypoint checkpoint are not supported.

See the [CVAT native-functions guide](https://docs.cvat.ai/docs/annotation/auto-annotation/ai-models/#native-functions)
for authentication, workspace scopes, and registration details.
