# RF-DETR

This is an implementation of a CVAT auto-annotation function that uses models from the RF-DETR
family, as implemented in the Roboflow `rfdetr` library.

The function code is provided under the MIT license. This function supports nondeprecated models
distributed under the Apache-2.0 license in `rfdetr==1.9.2`.
RF-DETR Plus and deprecated architectures are not supported.

The AA function supports models solving the following tasks:

- object detection
- instance segmentation
- pose estimation

The following pretrained models are supported:

| Task | Architecture names |
| --- | --- |
| Object detection | `RFDETRNano`, `RFDETRSmall`, `RFDETRMedium`, `RFDETRLarge` |
| Instance segmentation | `RFDETRSegNano`, `RFDETRSegSmall`, `RFDETRSegMedium`, `RFDETRSegLarge`, `RFDETRSegXLarge`, `RFDETRSeg2XLarge` |
| Pose estimation | `RFDETRKeypointPreview` (preview model) |

To use this with CVAT CLI, use the following options:

```text
--function-file func.py -p architecture=str:<architecture>
```

where `<architecture>` is one of the RF-DETR class names above, such as `RFDETRSegSmall`.
The `architecture` parameter is required. By default, RF-DETR automatically downloads and uses
that architecture's pretrained weights.

RF-DETR checkpoints do not always contain architecture metadata. Some official Roboflow checkpoints,
including Large, SegLarge, and KeypointPreview, lack the `args` and `model_name` fields needed for
automatic architecture detection. Requiring `architecture` makes loading explicit for both pretrained
and custom checkpoints, even when the file is renamed.

This function also supports the following options:

- `-p model=str:<path>` - optional path to an existing local checkpoint, replacing the default
  pretrained weights. For example:

  ```text
  --function-file func.py -p architecture=str:RFDETRSegSmall -p model=str:/path/to/best.pth
  ```

  The checkpoint must match the selected architecture. Saved training configuration is restored
  from checkpoint metadata when available. Custom checkpoints must provide unique class names.

- `-p device=str:<device>` - the PyTorch device, such as `cuda` or `cuda:0`, on which to run the model.
  By default, `cpu` is used. GPU inference requires compatible CUDA-enabled PyTorch dependencies.

- `-p keypoint_names_path=str:<path>` - path to a file with names of keypoints.
  Only valid for pose estimation models.
  By default, the pretrained human-pose model uses the 17 COCO keypoint names
  (`nose`, `left_eye`, `right_eye`, etc.). Custom keypoint checkpoints require this option
  and must contain their object class names; the wrapper does not infer a schema from the keypoint count.

  The `<path>` can point to a text file with one keypoint name per line, in the model's keypoint order.

  For different schemas per class, use a `.json` file mapping every class name to an ordered list
  of keypoint names, for example `{"cat": ["nose", "tail"], "car": ["front", "rear", "top"]}`.
  Names must be nonempty and unique within each class, and counts must match the checkpoint.
  Classes without keypoints in a mixed detection/keypoint checkpoint are not supported.

Segmentation models return masks, or polygons when mask-to-polygon conversion is requested.
Pose models return skeletons and require matching skeleton labels in CVAT.
The confidence threshold defaults to `0.5` when none is supplied by CVAT;
keypoints with confidence below `0.5` are marked outside.

Use the same architecture, optional checkpoint, and other parameters when registering the function
and starting the agent.
See the [native-functions guide](https://docs.cvat.ai/docs/annotation/auto-annotation/ai-models/#native-functions)
for dependency installation, authentication, and registration commands.
