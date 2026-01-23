"""
--- CNN with 10 classification heads
Accuracy: 86.30%
- Train loss: 0.08
- Epoch: 20
- Conv features: 3 -> 32 -> 64 -> 128
- Linear layer: 18432 -> 512
- Classifiers: 512 -> 62
"""

from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import string
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from tracker import Tracker

DATASET_DIR = "data/var_words/"
IMG_SIZE = 80
BATCH_SIZE = 16
LETTERS = string.ascii_letters + string.digits
BLANK_TOKEN = len(LETTERS)
LEN_LETTERS = len(LETTERS) + 1
NUM_LETTERS = 10
EPOCHS = 20


class SplitHorizontal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        split_w = W // NUM_LETTERS
        x = x.reshape(B, C, H, NUM_LETTERS, split_w)
        x = x.permute(0, 3, 1, 2, 4)
        x = x.reshape(B, NUM_LETTERS, -1)
        return x


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
        max_w = NUM_LETTERS * 16
        img = Image.new(img_raw.mode, (max_w, h), (255, 255, 255))
        img.paste(img_raw, (0, 0))
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img).transpose(2, 0, 1)
        img = torch.tensor(img, dtype=torch.float32) / 255.0

        label = torch.full((NUM_LETTERS,), BLANK_TOKEN, dtype=torch.long)
        chars_to_idx = {v: i for (i, v) in enumerate(LETTERS)}
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


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ),
            SplitHorizontal(),
            nn.Linear(128 * 2 * 20, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, LEN_LETTERS)
                )
                for _ in range(NUM_LETTERS)
            ]
        )

    def forward(self, x):
        features = self.features(x)
        B, S, F = features.shape  # Batch, Split, Features
        return torch.stack(
            [
                classifier(features[:, i, :].reshape(B, F))
                for (i, classifier) in enumerate(self.classifiers)
            ],
            dim=1,
        )


run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
t = Tracker(run_name=run_name)

model = Classifier()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Register model and optimizer for checkpointing
t.register("model", model)
t.register("optimizer", optimizer)

for epoch in range(EPOCHS):
    pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", total=len(train_loader))
    for i, (imgs, labels) in pbar:
        B, N = labels.shape
        out = model(imgs)  # B, N, C
        out = out.reshape(B * N, -1)  # B x N, C
        labels = labels.reshape(B * N)  # B x N

        optimizer.zero_grad()

        loss = F.cross_entropy(out, labels)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            pbar.set_postfix_str(f"Loss: {loss.item():.2f}")
            step = i + epoch * len(train_loader)
            t.log(step, train_loss=loss.item())
            t.plot("train_loss")

    correct_letters = 0
    total_letters = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        model.eval()
        for imgs, labels in tqdm(val_loader, desc="Validation"):
            logits = model(imgs)  # B, N, C
            preds = torch.argmax(logits, dim=2)  # B, N

            B, N = labels.shape
            mask = labels != BLANK_TOKEN  # B, N

            # Letter accuracy: count correct non-blank predictions
            correct_letters += ((preds == labels) & mask).sum()
            total_letters += mask.sum()

            # Word accuracy: all non-blank positions must be correct
            # For each sample, check if (correct OR is_blank) for ALL positions
            word_correct = ((preds == labels) | ~mask).all(dim=1)  # B,
            total_correct += word_correct.sum()
            total_samples += B

        print(f"Total correct: {total_correct}")
        print(f"Total samples: {total_samples}")
        print(
            f"Letter acc: {correct_letters * 100 / total_samples / NUM_LETTERS:.2f}%"
            f" ({correct_letters}/{total_samples * NUM_LETTERS})"
        )

        accuracy = total_correct * 100 / total_samples
        t.log(epoch, val_accuracy=accuracy)
        t.plot("val_accuracy")
        print(f"Accuracy: {accuracy:.3f}%")

        # Save checkpoint
        t.save_logs()
        t.save(epoch=epoch, is_best=False, keep_last=3)

        model.train()
