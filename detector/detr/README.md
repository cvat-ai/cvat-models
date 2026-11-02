# RF-DETR

This is an implementation of a CVAT auto-annotation function that uses models from the RF-DETR
family, as implemented in the Roboflow `rfdetr` library.

The function code is provided under the MIT license. This function supports nondeprecated models
distributed under the Apache-2.0 license in `rfdetr`.
RF-DETR Plus and deprecated architectures are not supported.

The AA function supports models solving the following tasks:

- object detection
- instance segmentation
- keypoint detection

The following pretrained models are supported:

| Task | Architecture names |
| --- | --- |
| Object detection | `RFDETRNano`, `RFDETRSmall`, `RFDETRMedium`, `RFDETRLarge` |
| Instance segmentation | `RFDETRSegNano`, `RFDETRSegSmall`, `RFDETRSegMedium`, `RFDETRSegLarge`, `RFDETRSegXLarge`, `RFDETRSeg2XLarge` |
| Keypoint detection | `RFDETRKeypointPreview` (preview model) |

To use this with CVAT CLI, use the following options:

```text
--function-file func.py -p architecture=str:<architecture>
```

where `<architecture>` is one of the RF-DETR class names above, such as `RFDETRSegSmall`.
The `architecture` parameter is required. By default, RF-DETR automatically downloads and uses
that architecture's pretrained weights.

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
  Only supported for keypoint detection models. Required when using a custom checkpoint.

  The `<path>` can point to a text file with one keypoint name per line, in the model's keypoint order.
  These names will be applied to each class.

  To use different keypoint names per class, provide a `.json` file mapping every class name to an ordered list
  of keypoint names, for example `{"cat": ["nose", "tail"], "car": ["front", "rear", "top"]}`.

  Keypoint names must be nonempty and unique within one class, and counts must match the checkpoint.

Detection models return bounding boxes.
Segmentation models return masks, or polygons when mask-to-polygon conversion is requested.
Keypoint detection models return skeletons and require matching skeleton labels in CVAT.

The confidence threshold supplied by CVAT applies to both detections and individual keypoints.
Keypoints with confidence below the supplied threshold are marked outside.
When no threshold is supplied, RF-DETR uses its default for detections,
and keypoints are not filtered by confidence.

Use the same architecture, optional checkpoint, and other parameters when registering the function
and starting the agent.
See the [native-functions guide](https://docs.cvat.ai/docs/annotation/auto-annotation/ai-models/#native-functions)
for dependency installation, authentication, and registration commands.
