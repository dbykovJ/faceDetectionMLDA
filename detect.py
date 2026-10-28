# %% Import
import torch
import torchvision
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import functional as F

from detect_helpers import (
    ask_model,
    downscale,
    draw_boxes,
    merge_overlapping_boxes,
    scale_back,
)
from model import RETINA_SIZE, BorderBox, ModelResult
from net import Net


# %% Load Model
def load_model(path: str) -> Net:
    model = Net()
    state = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    return model


net = load_model("Net.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# %% Load Image
image_name = "image"

image_path = f"{image_name}.jpg"
input_max_side = 1024
target_image = read_image(image_path, mode=ImageReadMode.RGB)
target_image = F.rgb_to_grayscale(target_image, num_output_channels=1)
target_image = F.convert_image_dtype(target_image, torch.float32)
height, width = target_image.shape[-2:]
input_scale = input_max_side / max(height, width)
input_size = [max(1, round(height * input_scale)), max(1, round(width * input_scale))]
target_image = F.resize(target_image, input_size, antialias=True)
target_image = F.normalize(target_image, mean=[0.5], std=[0.5])

# %% Detect faces
results: list[ModelResult] = []
step_size: int = 6
scaling_sizes: list[float] = [0.6, 0.4, 0.2, 0.1]

for downscaled in downscale(target_image, scaling_sizes):
    downscaled.image = downscaled.image.to(device)
    scaled_step = max(1, round(step_size * downscaled.scale))

    for xstep in range(0, downscaled.width - RETINA_SIZE + 1, scaled_step):
        for ystep in range(0, downscaled.height - RETINA_SIZE + 1, scaled_step):
            result: ModelResult | None = ask_model(downscaled, xstep, ystep, net)

            if result is not None:
                results.append(result)

# %% post processcaled_step
out_image = Image.open(image_path).resize(
    (input_size[1], input_size[0]), resample=Image.Resampling.LANCZOS
)

original_boxes: list[BorderBox] = scale_back(results)
print(f"Found {len(original_boxes)} boxes before merging.")
certainity_filter = 0.95
original_certain = [box for box in original_boxes if box.certainity > certainity_filter]
print(f"Found {len(original_certain)} boxes after merging.")


image_raw_filtered = draw_boxes(out_image, original_certain)
image_raw_filtered.save(f"{image_name}_raw_out.jpg")


def filter_by_support(boxes, min_neighbors=2, iou_thresh=0.2):
    t = torch.tensor(
        [[b.x, b.y, b.x + b.size, b.y + b.size] for b in boxes], dtype=torch.float32
    )
    iou = torchvision.ops.box_iou(t, t)
    counts = (iou > iou_thresh).sum(dim=1) - 1  # exclude self
    return [b for b, c in zip(boxes, counts.tolist()) if c >= min_neighbors]


boxes_by_support = filter_by_support(original_certain)

merged_boxes: list[BorderBox] = merge_overlapping_boxes(
    boxes_by_support, iou_threshold=0.1
)

print(
    f"Found {len(merged_boxes)} boxes with certainty > {certainity_filter} after filtering."
)

# %% Visualize and save

final_image = draw_boxes(out_image, merged_boxes)

final_image.save(f"{image_name}_out.jpg")
