from dataclasses import dataclass
from unittest import result

import cv2
import numpy
import torch
from PIL import Image

"""
hashmap of image scale and bonding box

for each scaling size do
   downscale the image
   for each step do:
     ask model for current window
     remember result for window and scaling 

 scale back the boxes to original image size
 find overlapping boxes and merge to optimal box
 draw the boxes
"""
RETINA_SIZE = 36
results: map[tuple[float, int, int], bool] = {}
window_size: int = 36
step_size: int = 6
scaling_sizes: list[float] = [0.8, 0.6, 0.4]
image: Image = Image.open("image.jpg")


@dataclass
class ModelResult:
    x: int
    y: int
    scale: float


@dataclass
class ImageLayer:
    image: torch.Tensor
    scale: float

@dataclass
class BorderBox:
    x: int
    y: int
    size: int = RETINA_SIZE


def downscale(image_path, scales: list[float]) -> list[ImageLayer]:
    target_image = cv2.imread(image_path)
    target_image = torch.from_numpy(target_image)
    if not target_image:
        raise ValueError("Image not provided")
    if not scales:
        raise ValueError("Scales not provided")

    images = [ImageLayer(target_image, 1)]
    for scale in scales:
        layer = cv2.resize(images[-1].image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        layer = torch.from_numpy(layer)
        images.append(ImageLayer(layer, scale))

    return images


def ask_model(image, xstep: int, ystep: int, window_size: int) -> ModelResult:
    # sonny
    return NotImplementedError("ask_model function is not implemented")


def scale_back(results: list[ModelResult]) -> list[BorderBox]:
    # dima
    final_boxes = []
    for box in results:
        x = box.x * (1/box.scale)
        y = box.y * (1/box.scale)
        size = RETINA_SIZE * (1/box.scale)
        final_boxes.append(BorderBox(int(round(i))) for i in [x, y, size])
    return NotImplementedError("scale_back function is not implemented")


""""
Returns a list of merged bounding boxes from a list of overlapping bounding boxes.
Each bounding box is represented as a tuple of two tuples,
where the first tuple contains the coordinates of the top-left corner and 
the second tuple contains the coordinates of the bottom-right corner.
"""


def merge_overlapping_boxes(boxes: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    # emil and lorenz
    return NotImplementedError("merge_overlapping_boxes function is not implemented")


def draw_boxes(image, boxes: list[tuple[int, int]]):
    # santiago

    return NotImplementedError("draw_boxes function is not implemented")


for scale in scaling_sizes:
    downscaled = downscale(image, scale)

    for xstep in range(0, downscaled.width - window_size, step_size):
        for ystep in range(0, downscaled.height - window_size, step_size):
            is_image: ModelResult = ask_model(downscaled, xstep, ystep, window_size)
            results[(scale, xstep, ystep)] = is_image

original_boxes: list[tuple[int, int]] = scale_back(results, scaling_sizes)
merged_boxes: list[tuple[int, int]] = merge_overlapping_boxes(original_boxes)
image_with_boxes: Image = draw_boxes(image, merged_boxes)

