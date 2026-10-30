import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.ops import nms
from torchvision.transforms import functional as F

from model import RETINA_SIZE, BorderBox, ImageLayer, ModelResult
from net import Net


def downscale(target_image: torch.Tensor, scales: list[float]) -> list[ImageLayer]:

    height, width = target_image.size(dim=1), target_image.size(dim=2)
    images = [ImageLayer(target_image, 1, height, width)]
    for scale in scales:
        output_height = max(1, round(height * scale))
        output_width = max(1, round(width * scale))
        layer = F.resize(
            images[0].image,
            size=[output_height, output_width],
            antialias=True,
        )
        layer_height, layer_width = layer.size(dim=1), layer.size(dim=2)
        images.append(ImageLayer(layer, scale, layer_height, layer_width))

    del images[0]

    return images


def ask_model(
    image: ImageLayer, xstep: int, ystep: int, model: Net
) -> ModelResult | None:

    image_slice = image.image[
        :, ystep : ystep + RETINA_SIZE, xstep : xstep + RETINA_SIZE
    ]

    outputs = model(image_slice)
    probs = torch.softmax(outputs, dim=1)
    probability, predicted = torch.max(probs, 1)

    return (
        ModelResult(xstep, ystep, image.scale, probability.item())
        if predicted.item() == 1
        else None
    )


@torch.inference_mode()
def ask_model_batch(
    image: ImageLayer, positions: list[tuple[int, int]], model: Net, min_logit: float = 0.0
) -> list[ModelResult]:
    if not positions:
        return []

    patches = torch.stack(
        [image.image[:, y : y + RETINA_SIZE, x : x + RETINA_SIZE] for x, y in positions]
    )
    logits = model(patches).flatten().cpu()  # (N,1,1,1) -> (N,)
    scores = logits  # a single logit is already the face score

    return [
        ModelResult(x, y, image.scale, score)
        for (x, y), score in zip(positions, scores.tolist())
        if score > min_logit
    ]


def filter_by_support(boxes, certainty_filter: float, min_neighbors=2, iou_thresh=0.2):
    t = torch.tensor(
        [[b.x, b.y, b.x + b.size, b.y + b.size] for b in boxes], dtype=torch.float32
    )
    iou = torchvision.ops.box_iou(t, t)
    counts = (iou > iou_thresh).sum(dim=1) - 1  # exclude self
    return [b for b, c in zip(boxes, counts.tolist()) if
            b.certainty >= certainty_filter and c >= min_neighbors]


def scale_back(results: list[ModelResult]) -> list[BorderBox]:
    # dima
    final_boxes = []
    for res in results:
        x = res.x * (1 / res.scale)
        y = res.y * (1 / res.scale)
        size = int(round(RETINA_SIZE * (1 / res.scale)))

        final_boxes.append(
            BorderBox(int(round(x)), int(round(y)), res.probability, size)
        )
    return final_boxes


""""
Returns a list of merged bounding boxes from a list of overlapping bounding boxes.
Each bounding box is represented as a tuple of two tuples,
where the first tuple contains the coordinates of the top-left corner and 
the second tuple contains the coordinates of the bottom-right corner.
"""


def merge_overlapping_boxes(
    boxes: list[BorderBox], iou_threshold: float = 0.5
) -> list[BorderBox]:
    # emil and lorenz

    if not boxes:
        return []

    flattened = np.array(
        [[box.x, box.y, box.x + box.size, box.y + box.size] for box in boxes]
    )

    tensor_boxes = torch.tensor(data=flattened, dtype=torch.float32)
    tensor_scores = torch.tensor(
        data=[box.certainty for box in boxes], dtype=torch.float32
    )

    res = nms(tensor_boxes, tensor_scores, iou_threshold)
    original_boxes = list(boxes)
    return [original_boxes[i] for i in res.tolist()]


def draw_boxes(image: Image.Image, boxes: list[BorderBox]) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)

    for box in boxes:
        draw.rectangle(
            [box.x, box.y, box.x + box.size, box.y + box.size], outline="green", width=2
        )

    return image
