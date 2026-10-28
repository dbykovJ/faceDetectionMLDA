# %% Import
import torch
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
image_path = "image.jpeg"
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
scaling_sizes: list[float] = [0.8, 0.6, 0.4, 0.2]

for downscaled in downscale(target_image, scaling_sizes):
    downscaled.image = downscaled.image.to(device)

    for xstep in range(0, downscaled.width - RETINA_SIZE, step_size):
        for ystep in range(0, downscaled.height - RETINA_SIZE, step_size):
            result: ModelResult | None = ask_model(downscaled, xstep, ystep, net)

            if result is not None:
                results.append(result)

# %% post process
original_boxes: list[BorderBox] = scale_back(results)
print(f"Found {len(original_boxes)} boxes before merging.")

merged_boxes: list[BorderBox] = merge_overlapping_boxes(
    original_boxes, iou_threshold=0.1
)

print(f"Found {len(merged_boxes)} boxes after merging.")

certainity_filter = 0.95
merged_boxes = [box for box in merged_boxes if box.certainity > certainity_filter]

print(
    f"Found {len(merged_boxes)} boxes with certainty > {certainity_filter} after filtering."
)


# %% Visualize and save
out_image = Image.open(image_path).resize(
    (input_size[1], input_size[0]), resample=Image.Resampling.LANCZOS
)

final_image = draw_boxes(out_image, merged_boxes)

final_image.save("output_image.jpg")
