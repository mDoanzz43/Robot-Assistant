from PIL import Image
import os

input_dir = r"D:\STUDY\At_school\asr_zipformer\robot_expressions"
output_dir = "data"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".png"):
        img = Image.open(os.path.join(input_dir, file)).resize((320,240))
        img = img.convert("RGB")

        raw = []
        for pixel in img.getdata():
            r, g, b = pixel
            rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
            raw.append(rgb565 & 0xFF)
            raw.append((rgb565 >> 8) & 0xFF)

        with open(os.path.join(output_dir, file.replace(".png",".raw")), "wb") as f:
            f.write(bytearray(raw))

print("Done convert!")