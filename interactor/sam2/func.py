# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import dataclasses

import cvat_sdk.auto_annotation as cvataa
import PIL.Image
import torch
from cvat_sdk.masks import encode_mask
from transformers import Sam2Processor, Sam2Model

@dataclasses.dataclass(frozen=True, kw_only=True)
class _PreprocessedImage:
    image_embeddings: list[torch.Tensor]
    original_sizes: torch.Tensor

class _Sam2Detector:
    spec = cvataa.InteractionFunctionSpec(min_pos_points=0, min_neg_points=0, min_bounding_boxes=0)

    def __init__(self, model_id: str, device: str = "cpu", **kwargs) -> None:
        self._processor = Sam2Processor.from_pretrained(model_id)
        self._model = Sam2Model.from_pretrained(model_id, device_map=device, **kwargs)

        if self._model.device.type == "cuda":
            torch.set_autocast_enabled(True)
            torch.set_autocast_gpu_dtype(torch.bfloat16)
            if torch.cuda.get_device_properties(self._model.device).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    @torch.inference_mode()
    def preprocess_image(
        self, context: cvataa.InteractionFunctionContext, image: PIL.Image.Image
    ) -> _PreprocessedImage:
        image = image.convert("RGB")

        inputs = self._processor(images=image, return_tensors="pt").to(self._model.device)
        image_embeddings = self._model.get_image_embeddings(inputs["pixel_values"])

        return _PreprocessedImage(
            image_embeddings=image_embeddings, original_sizes=inputs["original_sizes"]
        )

    @torch.inference_mode()
    def detect(
        self, context: cvataa.InteractionFunctionContext, pp_image: _PreprocessedImage,
        prompts: cvataa.InteractionPrompts,
    ) -> list[cvataa.InteractionResultShape]:
        if prompts.pos_points or prompts.neg_points:
            input_points = [[[*map(list, prompts.pos_points), *map(list, prompts.neg_points)]]]
            input_labels = [[[1] * len(prompts.pos_points) + [0] * len(prompts.neg_points)]]
        else:
            input_points = input_labels = None

        if prompts.bounding_box:
            input_boxes = [[[*prompts.bounding_box[0], *prompts.bounding_box[1]]]]
        else:
            input_boxes = None

        inputs = self._processor(
            original_sizes=pp_image.original_sizes,
            input_boxes=input_boxes,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt"
        ).to(self._model.device)

        outputs = self._model(
            image_embeddings=pp_image.image_embeddings, multimask_output=False, **inputs
        )

        masks = self._processor.post_process_masks(
            outputs.pred_masks.cpu(), pp_image.original_sizes,
        )[0][0]

        return [
            cvataa.InteractionResultShape(type="mask", points=encode_mask(mask))
            for mask in masks
        ]

create = _Sam2Detector
