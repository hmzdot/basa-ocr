import os
import random
import string
import shutil
from PIL import Image, ImageDraw, ImageFont
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

if os.path.exists("data/even_words"):
    if input("data/even_words already exists. Overwrite? (y/n): ") == "y":
        shutil.rmtree("data/even_words")

os.makedirs("data/even_words", exist_ok=True)
for i in tqdm(range(N), total=N):
    font_file = random.choice(font_files)
    font_path = f"data/fonts/{font_file}"
    font = ImageFont.truetype(font_path, 24)

    text = "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(NUM_LETTERS)
    )

    bg_color = random.choice(BACKGROUND_COLORS)
    fg_color = random.choice(FOREGROUND_COLORS)

    image = Image.new(mode="RGB", size=(128, 48), color=bg_color)
    draw = ImageDraw.Draw(image)

    # Split horizontal space into 5 equal parts
    section_width = 128 / NUM_LETTERS

    # Calculate vertical center (using first letter as reference for height)
    first_char_bbox = draw.textbbox((0, 0), text[0], font=font)
    char_height = first_char_bbox[3] - first_char_bbox[1]
    y = (48 - char_height) // 2

    # Place each letter in the center of its own section
    for idx, char in enumerate(text):
        # Calculate the center of this letter's section
        section_center_x = (idx + 0.5) * section_width

        # Calculate the bounding box for this individual character
        char_bbox = draw.textbbox((0, 0), char, font=font)
        char_width = char_bbox[2] - char_bbox[0]

        # Center the character within its section
        x = int(section_center_x - char_width / 2)

        draw.text((x, y), char, font=font, fill=fg_color)

    image.save(f"data/even_words/{i:05d}_{text}.png")
