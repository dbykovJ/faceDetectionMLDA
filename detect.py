from dataclasses import dataclass

from PIL import Image
from PIL.ImageFile import ImageFile

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


@dataclass
class ModelResult:
    probability: float
    is_face: bool


results: dict[tuple[float, int, int], ModelResult] = {}
window_size: int = 36
step_size: int = 6
scaling_sizes: list[float] = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
image = Image.open("image.jpg")


def downscale(image, scale: float) -> ImageFile:
    # dima
    raise NotImplementedError("downscale function is not implemented")


def ask_model(image, xstep: int, ystep: int, window_size: int) -> ModelResult:
    # sonny
    raise NotImplementedError("ask_model function is not implemented")


def scale_back(
    results: map[tuple[float, int, int], bool], scaling_sizes: list[float]
) -> list[tuple[int, int]]:
    # dima
    raise NotImplementedError("scale_back function is not implemented")


""""
Returns a list of merged bounding boxes from a list of overlapping bounding boxes.
Each bounding box is represented as a tuple of two tuples,
where the first tuple contains the coordinates of the top-left corner and 
the second tuple contains the coordinates of the bottom-right corner.
"""


def merge_overlapping_boxes(
    boxes: list[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    # emil and lorenz

    pass


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
