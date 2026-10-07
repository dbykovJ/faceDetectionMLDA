
// hashmap of image scale and bonding box

// for each scaling size do
//   downscale the image
//   for each step do:
//     ask model for current window
//     remember result for window and scaling 
// 
// scale back the boxes to original image size
// find overlapping boxes and merge to optimal box
// draw the boxes

results: map[tuple[float, int, int], bool] = {}
window_size: int = 36
step_size: int = 6

def downscale(image, scale: float) -> Image:
    return NotImplementedError("downscale function is not implemented")

def ask_model(image, xstep: int, ystep: int, window_size: int) -> bool:
    return NotImplementedError("ask_model function is not implemented")

def scale_back(results: map[tuple[float, int, int], bool], scaling_sizes: list[float]) -> list[tuple[int, int]]:
    return NotImplementedError("scale_back function is not implemented")

def merge_overlapping_boxes(boxes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return NotImplementedError("merge_overlapping_boxes function is not implemented")

def draw_boxes(image, boxes: list[tuple[int, int]]):
    return NotImplementedError("draw_boxes function is not implemented")

for scale in scaling_sizes:
    downscaled = downscale(image, scale)

    for xstep in range(0, downscaled.with - window_size, step_size):
        for ystep in range(0, downscaled.height - window_size, step_size):
            is_image: bool = ask_model(downscaled, xstep, ystep, window_size)
            results[(scale, xstep, ystep)] = is_image

original_boxes: list[tuple[int, int]]  = scale_back(results, scaling_sizes)
merged_boxes: list[tuple[int, int]] = merge_overlapping_boxes(original_boxes)
draw_boxes(image, merged_boxes)

