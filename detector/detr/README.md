# RF-DETR

This is an implementation of a CVAT auto-annotation function that uses models from the RF-DETR
family, as implemented in the Roboflow `rfdetr` library.

The function code is provided under the MIT license. This function supports nondeprecated models
distributed under the Apache-2.0 license in `rfdetr==1.9.2`.
RF-DETR Plus and deprecated architectures are not supported, including in local checkpoints.

The AA function supports models solving the following tasks:

- object detection
- instance segmentation
- pose estimation

The following pretrained models are supported:

| Task | Model names |
| --- | --- |
| Object detection | `RFDETRNano`, `RFDETRSmall`, `RFDETRMedium`, `RFDETRLarge` |
| Instance segmentation | `RFDETRSegNano`, `RFDETRSegSmall`, `RFDETRSegMedium`, `RFDETRSegLarge`, `RFDETRSegXLarge`, `RFDETRSeg2XLarge` |
| Pose estimation | `RFDETRKeypointPreview` (preview model) |

To use this with CVAT CLI, use the following options:

```text
--function-file func.py -p model=str:<model>
```

where `<model>` is one of the model names above, such as `RFDETRLarge`, or the path to a local
checkpoint file. Named models automatically download their pretrained weights.
For a local checkpoint, RF-DETR must be able to infer its architecture from the checkpoint metadata.
Custom checkpoints must use a supported architecture and provide unique class names.

This function also supports the following options:

- `-p device=str:<device>` - the PyTorch device, such as `cuda` or `cuda:0`, on which to run the model.
  By default, `cpu` is used. GPU inference requires compatible CUDA-enabled PyTorch dependencies.

- `-p pretrain_weights=str:<path>` - path to a local checkpoint for a named model.
  For example, combine `-p model=str:RFDETRLarge` with `-p pretrain_weights=str:/path/to/checkpoint.pth`.
  The checkpoint architecture must match the selected model. Use this option for weight files
  without architecture metadata. It cannot be combined with a checkpoint path in `model`.

- `-p keypoint_names_path=str:<path>` - path to a file with names of keypoints.
  Only valid for pose estimation models.
  By default, the pretrained human-pose model uses the 17 COCO keypoint names
  (`nose`, `left_eye`, `right_eye`, etc.). Custom schemas require this option.

  The `<path>` can point to a text file, with one keypoint name per line. Leading and trailing
  whitespace will be ignored, and so will empty lines. The list applies to every class.

  For different schemas per class, use a `.json` file mapping every class name to an ordered list
  of keypoint names, for example `{"cat": ["nose", "tail"], "car": ["front", "rear", "top"]}`.
  Names must be nonempty and unique within each class, and counts must match the checkpoint.
  Classes without keypoints in a mixed detection/keypoint checkpoint are not supported.

Segmentation models return masks, or polygons when mask-to-polygon conversion is requested.
Pose models return skeletons and require matching skeleton labels in CVAT.
The confidence threshold defaults to `0.5` when none is supplied by CVAT;
keypoints with confidence below `0.5` are marked outside.

Use the same model parameters when registering the function and starting the agent.
See the [native-functions guide](https://docs.cvat.ai/docs/annotation/auto-annotation/ai-models/#native-functions)
for dependency installation, authentication, and registration commands.
