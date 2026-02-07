import io
import string
import torch
import torch.nn.functional as F
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader

BATCH_SIZE = 16
TARGET_HEIGHT = 32
DATASET_NAME = "priyank-m/MJSynth_text_recognition"


def resize_to_height(img, target_height=TARGET_HEIGHT):
    if isinstance(img, torch.Tensor):
        h, w = img.shape[-2], img.shape[-1]
    else:
        w, h = img.size
    new_w = max(1, int(round(w * (target_height / h))))
    return TF.resize(img, [target_height, new_w], interpolation=InterpolationMode.BILINEAR)


transform = transforms.Compose([transforms.Lambda(resize_to_height), transforms.ToTensor()])

# Keep raw bytes/path from HF instead of auto-decoding via PIL in the iterator.
# This prevents a single corrupted image from terminating the whole stream.
train_dataset = load_dataset(DATASET_NAME, split="train", streaming=True).cast_column(
    "image",
    HFImage(decode=False),
)
val_dataset = load_dataset(DATASET_NAME, split="test", streaming=True).cast_column(
    "image",
    HFImage(decode=False),
)


class Vocabulary:
    def __init__(self):
        self.chars = string.ascii_letters + string.digits

        self.blank_token = 0
        self.char_to_idx = {ch: i + 1 for (i, ch) in enumerate(self.chars)}
        self.idx_to_char = {i + 1: ch for (i, ch) in enumerate(self.chars)}
        self.idx_to_char[0] = "<blank>"

    def __len__(self):
        return len(self.chars) + 1

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        return "".join([self.idx_to_char[idx] for idx in indices if idx != 0])


vocab = Vocabulary()


_COLLATE_SKIP_LOGS = {"count": 0}


def _decode_item_image(image_obj):
    if isinstance(image_obj, Image.Image):
        return image_obj

    if isinstance(image_obj, dict):
        image_bytes = image_obj.get("bytes")
        image_path = image_obj.get("path")
        if image_bytes is not None:
            with Image.open(io.BytesIO(image_bytes)) as img:
                return img.convert("RGB")
        if image_path:
            with Image.open(image_path) as img:
                return img.convert("RGB")
    raise ValueError("Unsupported image payload")


def collate_fn(items):
    images = []
    labels = []

    for item in items:
        try:
            image = _decode_item_image(item["image"])
            image = transform(image)
        except (OSError, ValueError, RuntimeError, UnidentifiedImageError) as err:
            _COLLATE_SKIP_LOGS["count"] += 1
            if _COLLATE_SKIP_LOGS["count"] <= 10:
                print(f"[collate] skipping bad sample due to decode error: {err}")
            continue

        images.append(image)
        labels.append(item["label"])

    if not images:
        return None

    image_widths = torch.tensor([img.shape[2] for img in images])
    max_width = image_widths.max().item()
    images = torch.stack(
        [
            F.pad(image, (0, max_width - image.size(2)), "constant", 0)
            for image in images
        ],
        dim=0,
    )

    label_lengths = torch.tensor([len(label) for label in labels])
    max_label = int(label_lengths.max().item())
    labels = [vocab.encode(label) + [0] * (max_label - len(label)) for label in labels]
    labels = torch.stack([torch.tensor(label) for label in labels])

    return {
        "images": images,
        "image_widths": image_widths,
        "labels": labels,
        "label_lengths": label_lengths,
    }


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

len_train = 7_220_000 // BATCH_SIZE
len_val = 803_000 // BATCH_SIZE
