from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from torchvision.transforms import functional as F

from detect_helpers import (
    ask_model_batch,
    downscale,
    draw_boxes,
    filter_by_certainty,
    filter_by_neighbors,
    merge_overlapping_boxes,
    scale_back,
)
from model import RETINA_SIZE, BorderBox, ModelResult


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
    DEFAULT_THICKNESS = 2

    def __init__(
        self,
        net: torch.nn.Module,
        input_max_side: int = DEFAULT_SIZE,
        step_size: int = DEFAULT_STEP,
        scaling_sizes: tuple[float, ...] = DEFAULT_SCALING_SIZES,
        certainty_filter: float = DEFAULT_CERTAINTY,
        knn: int = DEFAULT_KNN,
        thickness: int = DEFAULT_THICKNESS,
        debug: bool = False,
    ):
        if thickness < 1:
            raise ValueError("Box thickness must be at least 1")
        self.thickness = thickness
        self.debug = debug
        self.input_max_side = input_max_side
        self.step_size = step_size
        self.scaling_sizes = scaling_sizes
        self.certainty_filter = certainty_filter
        self.knn = knn
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
        started = perf_counter()
        img = Image.open(image_path)
        original_size = img.size
        img = self._resize_image(img)
        device = next(self.net.parameters()).device
        print(
            f"Detecting {image_path}: {original_size[0]}x{original_size[1]} -> "
            f"{img.width}x{img.height}, device={device}, window={RETINA_SIZE}x{RETINA_SIZE}",
            flush=True,
        )

        img_tensor = self._to_normalized_tensor(img)

        results: list[ModelResult] = []
        total_windows = 0
        for index, downscaled in enumerate(downscale(img_tensor, self.scaling_sizes), 1):
            downscaled.image = downscaled.image.to(device)
            scaled_step = max(1, round(self.step_size * downscaled.scale))
            x_positions = range(0, downscaled.width - RETINA_SIZE + 1, scaled_step)
            y_positions = range(0, downscaled.height - RETINA_SIZE + 1, scaled_step)
            window_count = len(x_positions) * len(y_positions)
            total_windows += window_count
            print(
                f"  Scale {index}/{len(self.scaling_sizes)} ({downscaled.scale:.4g}): "
                f"{downscaled.width}x{downscaled.height}, stride={scaled_step}, "
                f"{window_count:,} windows to check"
                + (" (image smaller than detection window; skipped)" if not window_count else ""),
                flush=True,
            )
            positions: list[tuple[int, int]] = []

            for xstep in x_positions:
                for ystep in y_positions:
                    positions.append((xstep, ystep))
                    if len(positions) == batch_size:
                        results.extend(ask_model_batch(downscaled, positions, self.net))
                        positions.clear()

            results.extend(ask_model_batch(downscaled, positions, self.net))

        boxes: list[BorderBox] = scale_back(results)

        if self.debug:
            draw_boxes(img, boxes, self.thickness).save("before_filtering.jpg")

        boxes_by_certainty = filter_by_certainty(boxes, self.certainty_filter)

        if self.debug:
            draw_boxes(img, boxes_by_certainty, self.thickness).save("after_certainty.jpg")
        boxes_by_neighbors = filter_by_neighbors(
            boxes_by_certainty,
            min_neighbors=self.knn,
            iou_thresh=0.2,
        )
        if self.debug:
            draw_boxes(img, boxes_by_neighbors, self.thickness).save("after_neighbors.jpg")

        merged_boxes: list[BorderBox] = merge_overlapping_boxes(
            boxes_by_neighbors, iou_threshold=0.1
        )
        print(
            f"  Checked {total_windows:,} windows: {len(boxes):,} candidates -> "
            f"{len(boxes_by_certainty):,} after certainty "
            f"({self.certainty_filter:g}) -> {len(boxes_by_neighbors):,} after neighbors "
            f"(minimum {self.knn}) -> {len(merged_boxes):,} final boxes. "
            f"Elapsed: {perf_counter() - started:.2f}s",
            flush=True,
        )

        return draw_boxes(img, merged_boxes, self.thickness)


# TODO - recreate images predicted from the plain sliding window approach (without knn filtering) {also show the boxes
# before merging overlapping boxes}, then with knn filtering, then show the results of 3 different models
# also show different scaling sizes combinations
# show the certainty filter tuning
# both of these are shown on the simple model
