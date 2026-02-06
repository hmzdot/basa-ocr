import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, IterableDataset

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


transform = transforms.Compose(
    [
        transforms.Lambda(resize_to_height),
        transforms.ToTensor(),
    ]
)

class SafeStreamingDataset(IterableDataset):
    def __init__(self, dataset, split_name, max_skip_logs=5):
        self.dataset = dataset
        self.split_name = split_name
        self.max_skip_logs = max_skip_logs

    def __iter__(self):
        it = iter(self.dataset)
        skipped = 0
        while True:
            try:
                sample = next(it)
            except StopIteration:
                if skipped:
                    print(f"[{self.split_name}] skipped {skipped} bad samples")
                break
            except (OSError, ValueError, RuntimeError) as err:
                skipped += 1
                if skipped <= self.max_skip_logs:
                    print(
                        f"[{self.split_name}] skipping bad sample due to decode error: {err}",
                    )
                continue
            yield sample


train_dataset = SafeStreamingDataset(
    load_dataset(DATASET_NAME, split="train", streaming=True),
    split_name="train",
)
val_dataset = SafeStreamingDataset(
    load_dataset(DATASET_NAME, split="test", streaming=True),
    split_name="val",
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


def collate_fn(items):
    images = [transform(item["image"]) for item in items]

    image_widths = torch.tensor([img.shape[2] for img in images])
    max_width = image_widths.max().item()
    images = torch.stack(
        [
            F.pad(image, (0, max_width - image.size(2)), "constant", 0)
            for image in images
        ],
        dim=0,
    )

    labels = [item["label"] for item in items]
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
