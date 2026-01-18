"""
Initial accuracy: 0.00%
- Train loss: 4.11 (resonated between 4.10-4.16)
- Epoch: 10
- Conv features: 3 -> 32 -> 64 -> 128
- Linear layer: 18432 -> 512
- Classifiers: 512 -> 62

Increase linear layer size: 0.00%
- Train loss: 4.13 (less resonating, but still bad convergence)
- Epoch: 10
- Conv features: 3 -> 32 -> 64 -> 128
* Linear layer: 18432 -> 2048 -> 512
- Classifiers: 512 -> 62

Increase classifier size: 0.00%
- Train loss: 4.13
- Epoch: 10
- Conv features: 3 -> 32 -> 64 -> 128
* Linear layer: 18432 -> 2048
* Classifiers: 2048 -> 512 -> 62

Add a ConvBlock: 0.00%
- Train loss: 4.13
- Epoch: 10
* Conv features: 3 -> 32 -> 64 -> 128 -> 256
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

from train_utils import Checkpointer, Tracker

IMG_SIZE = 60
BATCH_SIZE = 16
LETTERS = string.ascii_letters + string.digits
LEN_LETTERS = len(LETTERS)
NUM_LETTERS = 5
EPOCHS = 10


class SplitHorizontal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # torch.chunk(x, NUM_LETTERS, dim=3)
        B, C, H, W = x.shape
        split_w = W // NUM_LETTERS
        x = x.reshape(B, C, H, NUM_LETTERS, split_w)
        x = x.permute(0, 3, 1, 2, 4)
        x = x.reshape(B, NUM_LETTERS, -1)
        return x


class FixedWordsDataset(Dataset):
    image_list: list[str]

    def __init__(self):
        super().__init__()
        self.image_list = os.listdir("data/fixed_words")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_path = os.path.join("data/fixed_words", image_name)
        img = Image.open(image_path).resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img).transpose(2, 0, 1)
        img = torch.tensor(img, dtype=torch.float32) / 255.0

        chars_to_idx = {v: i for (i, v) in enumerate(LETTERS)}
        label_str = image_name.split("_")[1].split(".")[0]
        label = torch.tensor([chars_to_idx[l] for l in label_str])
        return img, label


dataset = FixedWordsDataset()
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
            nn.Linear(128 * 3 * 15, 2048),
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
c = Checkpointer(run_name=run_name)


model = Classifier()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

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
            t.add_stat("train_loss", i + epoch * len(train_loader), loss.item())
            t.save_plot("train_loss")
            t.save_txt("train_loss")

    correct_letters = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        model.eval()
        for imgs, labels in tqdm(val_loader, desc="Validation"):
            logits = model(imgs)  # B, N, C
            preds = torch.argmax(logits, dim=2)  # B, N
            correct_letters += (preds == labels).sum()
            total_correct += (preds == labels).all(dim=1).sum()
            total_samples += labels.shape[0]

        print(f"Total correct: {total_correct}")
        print(f"Total samples: {total_samples}")
        print(
            f"Letter acc: {correct_letters * 100 / total_samples / NUM_LETTERS:.2f}%"
            f" ({correct_letters}/{total_samples * NUM_LETTERS})"
        )

        accuracy = total_correct * 100 / total_samples
        t.add_stat("val_accuracy", epoch, accuracy)
        t.save_plot("val_accuracy")
        print(f"Accuracy: {accuracy:.3f}%")
        model.train()
