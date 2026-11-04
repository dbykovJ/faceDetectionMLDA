import numpy as np
from PIL import Image, ImageDraw

img = Image.open("images/image2.jpg").convert("RGB")
center = img.width / 2, img.height / 2
window_size = 36

geometric_scales = [0.5 * 0.8**i for i in range(10)]
linear_scales = np.linspace(geometric_scales[0], geometric_scales[-1], 8)
panels = [linear_scales, geometric_scales]

comparison = Image.new("RGB", (len(panels) * img.width, img.height))

for panel_index, scales in enumerate(panels):
    panel = img.copy()
    draw = ImageDraw.Draw(panel)

    for scale in scales:
        box_size = round(window_size / scale)
        x = round(center[0] - box_size / 2)
        y = round(center[1] - box_size / 2)
        draw.rectangle([x, y, x + box_size, y + box_size], outline="lime", width=2)

    offset = panel_index * img.width
    comparison.paste(panel, (offset, 0))

comparison.save("out/image2_scale_comparison.jpg")
