from PIL import Image
import numpy as np

w, h = 320, 240

with open(r"D:\STUDY\At_school\asr_zipformer\robot_expressions\data\angry.raw", "rb") as f:
    raw = np.frombuffer(f.read(), dtype=np.uint16)

# check size
if raw.size != w*h:
    raise ValueError(f"Wrong size: {raw.size}, expected {w*h}")

# convert RGB565 -> RGB888
img = np.zeros((h, w, 3), dtype=np.uint8)

for i in range(h * w):
    pixel = raw[i]
    r = ((pixel >> 11) & 0x1F) << 3
    g = ((pixel >> 5) & 0x3F) << 2
    b = (pixel & 0x1F) << 3
    img[i // w, i % w] = [r, g, b]

Image.fromarray(img).show()