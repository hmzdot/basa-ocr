"""
--- CRNN 

Initial
Accuracy: 86.60%
- Train loss: 0.03
- Epoch: 20
- Conv features: 3 -> 32 -> 64 -> 128
- LSTM hidden state: 256
- LSTM layers: 2

Deeper CNN
Accuracy: 86.60%
- Train loss: 0.03
- Epoch: 20
* Conv features: 3 -> 32 -> 64 -> 128 -> 256
- LSTM hidden state: 256
- LSTM layers: 2
"""

from datetime import datetime

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import string
import shutil
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from tracker import Tracker

DATASET_DIR = "data/var_words/"
IMAGE_HEIGHT = 32
BATCH_SIZE = 16
LETTERS = string.ascii_letters + string.digits
BLANK_TOKEN = 0
LEN_LETTERS = len(LETTERS) + 1
MAX_LETTERS = 10
EPOCHS = 20

class VarWordsDataset(Dataset):
    image_list: list[str]

    def __init__(self):
        super().__init__()
        self.image_list = os.listdir(DATASET_DIR)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_path = os.path.join(DATASET_DIR, image_name)

        # Pad the image from the right
        img_raw = Image.open(image_path)
        _w, h = img_raw.size
        max_w = MAX_LETTERS * 16
        img = Image.new(img_raw.mode, (max_w, h), (255, 255, 255))
        img.paste(img_raw, (0, 0))
        img = np.array(img).transpose(2, 0, 1)
        img = torch.tensor(img, dtype=torch.float32) / 255.0

        label = torch.full((MAX_LETTERS,), BLANK_TOKEN, dtype=torch.long)
        chars_to_idx = {v: (i+1) for (i, v) in enumerate(LETTERS)}
        label_str = image_name.split("_")[1].split(".")[0]
        label_ids = torch.tensor([chars_to_idx[l] for l in label_str])
        label[torch.arange(len(label_str))] = label_ids
        return img, label


dataset = VarWordsDataset()
len_train = int(len(dataset) * 0.8)
len_val = int(len(dataset) - len_train)
train_dataset, val_dataset = random_split(
    dataset, lengths=(len_train, len_val), generator=torch.Generator()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool: tuple[int, int] | None = None,
    ):
        super().__init__()
        modules = [
            nn.Conv2d(
                in_chans,
                out_chans,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
        ]
        if pool:
            modules.append(nn.MaxPool2d(kernel_size=pool, stride=pool))

        self.layer = nn.Sequential(*modules)

    def forward(self, x):
        return self.layer(x)

class CRNN(nn.Module):
    def __init__(
        self,
        height: int,
        num_classes: int,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
    ):
        super().__init__()
        # In: B, 3, H, W
        self.cnn = nn.Sequential(
            ConvLayer(3, 32, pool=(2,2)),       # B, 32, H/2, W/2
            ConvLayer(32, 64, pool=(2,2)),      # B, 64, H/4, W/4 
            ConvLayer(64, 128, pool=(2,1)),     # B, 128, H/8, W/4
            ConvLayer(128, 256, pool=(2,1)),    # B, 128, H/16, W/4
        )
        # Out: B, 128, H/16, W/4

        # In: B, W/4, 128 * H/16
        self.rnn = nn.LSTM(
            input_size=256 * (height//16),
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True
        )
        # Out: B, W/4, d_h * 2

        # In: B, W/4, d_h * 2
        self.transcription = nn.Linear(rnn_hidden * 2, num_classes)
        # Out: B, W/4, d_c

    def forward(self, x):
        x = self.cnn(x)

        B, C, H, W = x.shape

        x = x.permute(0, 3, 1, 2)       # B, W/8, 128, H/4
        x = x.reshape(B, W, -1)         # B, W/8, 128 * H/4 

        x, _ = self.rnn(x)              # B, W/8, d_h * 2

        x = self.transcription(x)       # B, W/8, d_c
        return x 

def ctc_greedy_decode(x: torch.Tensor) -> torch.Tensor:
    """
    In: x (B, T)
    Out: (B, max_len)
    """
    B, T = x.shape
    out = torch.full((B, MAX_LETTERS), BLANK_TOKEN, dtype=torch.long)
    
    for b in range(B):
        decoded = []
        prev_char = BLANK_TOKEN
        
        for t in range(T):
            cur_char = x[b, t].item()
            if cur_char != BLANK_TOKEN and cur_char != prev_char:
                decoded.append(cur_char)
            prev_char = cur_char
        
        for i, c in enumerate(decoded[:MAX_LETTERS]):
            out[b, i] = c
    
    return out

run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
t = Tracker(run_name=run_name)

model = CRNN(height=IMAGE_HEIGHT, num_classes=LEN_LETTERS)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Register model and optimizer for checkpointing
t.register("model", model)
t.register("optimizer", optimizer)

RUN_NAME = "2026-01-24_03-54-31"
t.load(RUN_NAME, which="epoch_0012")

total_correct = 0
total_samples = 0

with torch.no_grad():
    model.eval()

    incorrect = []
    for imgs, labels in tqdm(val_loader, desc="Validation"):
        logits = model(imgs)  # N, T, C
        preds = torch.argmax(logits, dim=2)  # N, T
        preds = ctc_greedy_decode(preds) # N, MAX_LETTERS

        B, N = labels.shape
        mask = labels != BLANK_TOKEN  # B, N

        incorrect_labels = ((preds != labels) & mask).any(dim=1)
        incorrect.extend(
            list(
                zip(
                    labels[incorrect_labels, :].cpu().detach().tolist(),
                    preds[incorrect_labels, :].cpu().detach().tolist(),
                )
            )
        )

        # Word accuracy: all non-blank positions must be correct
        # For each sample, check if (correct OR is_blank) for ALL positions
        word_correct = ((preds == labels) | ~mask).all(dim=1)  # B,
        total_correct += word_correct.sum()
        total_samples += B

    # Sample incorrect ones
    SAMPLE_SIZE = 20
    samples = random.choices(incorrect, k=SAMPLE_SIZE)
    os.makedirs(f"logs/{RUN_NAME}/failed", exist_ok=True)
    label_idx_to_char = {i + 1: c for i, c in enumerate(LETTERS)}
    label_to_images: dict[str, list[str]] = {}
    for image_name in dataset.image_list:
        label_str = image_name.split("_")[1].split(".")[0]
        label_to_images.setdefault(label_str, []).append(image_name)

    copied = 0
    for sample_label_ids, sample_pred_ids in samples:
        sample_label = "".join(
            label_idx_to_char[idx] for idx in sample_label_ids if idx != BLANK_TOKEN
        )
        sample_pred = "".join(
            label_idx_to_char[idx] for idx in sample_pred_ids if idx != BLANK_TOKEN
        )
        matches = label_to_images.get(sample_label, [])
        if not matches:
            continue
        image_name = matches.pop(0)
        src_path = os.path.join(DATASET_DIR, image_name)
        if sample_pred:
            base, ext = os.path.splitext(image_name)
            image_name = f"{base}_pred-{sample_pred}{ext}"
        dst_path = os.path.join(f"logs/{RUN_NAME}/failed", image_name)
        shutil.copy2(src_path, dst_path)
        copied += 1

    print(f"Incorrect samples: {len(incorrect)}")
    print(f"Copied failed samples: {copied}")

    print(f"Total correct: {total_correct}")
    print(f"Total samples: {total_samples}")

    accuracy = total_correct * 100 / total_samples
    print(f"Accuracy: {accuracy:.3f}%")
