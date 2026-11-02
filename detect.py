from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as F

from detect_helpers import (
    ask_model_batch,
    downscale,
    draw_boxes,
    filter_by_support,
    merge_overlapping_boxes,
    scale_back,
)
from model import RETINA_SIZE, BorderBox, ModelResult
from net import Net, load_model



class Detect:
    DEFAULT_SCALING_SIZES: tuple[float, ...] = (
        0.5,
        0.4,
        0.32,
        0.256,
        0.2048,
        0.16384,
        0.131072,
        0.1048576,
    )
    DEFAULT_STEP = 6
    DEFAULT_SIZE = 1024
    DEFAULT_CERTAINTY = 0.95
    DEFAULT_KNN = 2

    def __init__(
        self,
        net: torch.nn.Module,
        input_max_side: int = DEFAULT_SIZE,
        step_size: int = DEFAULT_STEP,
        scaling_sizes: tuple[float, ...] = DEFAULT_SCALING_SIZES,
        certainty_filter: float = DEFAULT_CERTAINTY,
        knn: int = DEFAULT_KNN,
    ):
        self.input_max_side = input_max_side
        self.step_size = step_size
        self.scaling_sizes = scaling_sizes
        self.certainty_filter = certainty_filter
        self.net = net

    def _resize_image(self, image: Image.Image):
        scale = self.input_max_side / max(image.height, image.width)
        return image.resize(
            (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            ),
            resample=Image.Resampling.LANCZOS,
        )

    def _to_normalized_tensor(self, image: Image.Image):
        target_image = F.to_tensor(image)
        target_image = F.rgb_to_grayscale(target_image, num_output_channels=1)
        target_image = F.convert_image_dtype(target_image, torch.float32)
        target_image = F.normalize(target_image, mean=[0.5], std=[0.5])
        return target_image

    def detect_faces(self, image_path: Path, batch_size=250):

        img = Image.open(image_path)
        img = self._resize_image(img)

        img_tensor = self._to_normalized_tensor(img)

        results: list[ModelResult] = []
        for downscaled in downscale(img_tensor, self.scaling_sizes):
            downscaled.image = downscaled.image.to(next(self.net.parameters()).device)
            scaled_step = max(1, round(self.step_size * downscaled.scale))
            positions: list[tuple[int, int]] = []

            for xstep in range(0, downscaled.width - RETINA_SIZE + 1, scaled_step):
                for ystep in range(0, downscaled.height - RETINA_SIZE + 1, scaled_step):
                    positions.append((xstep, ystep))
                    if len(positions) == batch_size:
                        results.extend(ask_model_batch(downscaled, positions, self.net))
                        positions.clear()

            results.extend(ask_model_batch(downscaled, positions, self.net))

        boxes: list[BorderBox] = scale_back(results)

        boxes_by_support = filter_by_support(
            boxes,
            certainty_filter=self.certainty_filter,
            min_neighbors=2,
            iou_thresh=0.2,
        )

        merged_boxes: list[BorderBox] = merge_overlapping_boxes(
            boxes_by_support, iou_threshold=0.1
        )

        return draw_boxes(img, merged_boxes)


# TODO - recreate images predicted from the plain sliding window approach (without knn filtering) {also show the boxes
# before merging overlapping boxes}, then with knn filtering, then show the results of 3 different models
# also show different scaling sizes combinations
# show the certainty filter tuning
# both of these are shown on the simple model
