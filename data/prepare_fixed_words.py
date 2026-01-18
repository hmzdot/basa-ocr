import os
import random
import string
import math
from PIL import Image, ImageDraw, ImageFont, ImageTransform
from tqdm import tqdm

N = 10000
NUM_LETTERS = 5

BACKGROUND_COLORS = [
    (255, 255, 255),
    (245, 245, 245),
    (255, 253, 208),
    (173, 216, 230),
    (144, 238, 144),
    (255, 182, 193),
]

FOREGROUND_COLORS = [
    (0, 0, 0),
    (64, 64, 64),
    (0, 0, 128),
    (139, 0, 0),
    (0, 100, 0),
    (75, 0, 130),
]

font_files = [f for f in os.listdir("data/fonts") if f.endswith(".ttf")]

for i in tqdm(range(N), total=N):
    font_file = random.choice(font_files)
    font_path = f"data/fonts/{font_file}"
    font = ImageFont.truetype(font_path, 24)

    text = "".join(random.choice(
        string.ascii_letters + string.digits
    ) for _ in range(NUM_LETTERS))

    bg_color = random.choice(BACKGROUND_COLORS)
    fg_color = random.choice(FOREGROUND_COLORS)

    image = Image.new(mode="RGB", size=(128, 48), color=bg_color)
    draw = ImageDraw.Draw(image)

    x_offset = 0
    y_offset = 0

    if random.random() < 0.2:
        x_offset = random.randint(-10, 10)
        y_offset = random.randint(-10, 10)

    draw.text((32 + x_offset, 10 + y_offset), text, font=font, fill=fg_color)

    if random.random() < 0.15:
        angle = random.uniform(-10, 10)
        image = image.rotate(angle, expand=False, fillcolor=bg_color)

    if random.random() < 0.1:
        x_shear = random.uniform(-0.2, 0.2)
        image = image.transform(
            (128, 48), ImageTransform.AffineTransform((1, x_shear, 0, 0, 1, 0))
        )

    os.makedirs("data/fixed_words", exist_ok=True)
    image.save(f"data/fixed_words/{i:02}_{text}.png")
